import copy
import random

import onnx
from onnx import shape_inference

import numpy as np

from mutation import utils, attr_gen, shape_utils


class Node:
    def __init__(self, op_type, num_in, in_edges=None):
        self.op_type = op_type
        self.num_in = num_in
        if in_edges:
            self.in_edges = list(copy.copy(in_edges))
        else:
            self.in_edges = []

    def get_op_type(self):
        return self.op_type

    def get_all_edges(self):
        return tuple(self.in_edges)

    def is_empty(self):
        return not self.in_edges

    def is_full(self):
        return self.num_in >= len(self.in_edges)

    def get_edge(self, i):
        return self.in_edges[i]

    def set_edge(self, i, e):
        self.in_edges[i] = e

    def add_edge(self, e):
        self.in_edges.append(e)

    def broadcast(self):
        # TODO: Replace if-else
        if self.op_type == 'Add' or self.op_type == 'Sub' or self.op_type == 'Mul':
            return True
        return False


class NodeGen:
    def __init__(self, next_node_idx, next_edge_idx):
        self.node_id = next_node_idx
        self.edge_id = next_edge_idx

    def new_node_name(self, node_type):
        return "%s_%d" % (node_type, self.node_id)

    def new_edge_name(self):
        return str(self.edge_id)

    @staticmethod
    def new_tensor_name(node_name, attr_name):
        return "%s_%s" % (node_name, attr_name)

    def new_node_with_tensor(self, node_type, attr_dict, *input_edges, **kwargs):
        node_name = self.new_node_name(node_type)
        attr_tensors = []
        attr_edges = []
        for attr_name, attr_val in attr_dict.items():
            attr_tensor = NodeGen.new_tensor(attr_val, node_name, attr_name)
            attr_tensors.append(attr_tensor)
            attr_edges.append(attr_tensor.name)
        node = self.new_node(node_type, *input_edges, *attr_edges, **kwargs)
        return node, attr_tensors

    def new_node_from_node_class(self, node):
        return self.new_node(
            node.get_op_type(), *[e.name for e in node.get_all_edges()]
        )

    def new_node_with_specified_output(self, node, output_edges):
        op_type = node.get_op_type()
        node = onnx.helper.make_node(
            op_type,
            [e.name for e in node.get_all_edges()],
            [e.name for e in output_edges],
            self.new_node_name(op_type)
        )
        self.node_id += 1
        return node

    def new_node(self, node_type, *input_edges, **kwargs):
        node = onnx.helper.make_node(
            node_type,
            input_edges,
            [self.new_edge_name()],
            self.new_node_name(node_type),
            **kwargs
        )
        self.node_id += 1
        self.edge_id += 1
        return node

    def new_edge(self, elem_type, tensor_shape):
        onnx_type = utils.numpy_onnx_type_mapping(elem_type)
        edge = onnx.helper.make_tensor_value_info(
            self.new_edge_name(), onnx_type, tensor_shape
        )
        self.edge_id += 1
        return edge

    @staticmethod
    def new_tensor(np_val, node_name, attr_name):
        data_type = utils.numpy_onnx_type_mapping(np_val.dtype)

        return onnx.helper.make_tensor(
            name=NodeGen.new_tensor_name(node_name, attr_name),
            data_type=data_type,
            dims=np_val.shape,
            vals=np_val.flatten()
        )


class GraphGen:
    def __init__(self, model):
        self.ins_idx = None
        self.model = shape_inference.infer_shapes(model)
        self.graph = self.model.graph
        max_node_idx = utils.get_max_node_idx(self.graph)
        max_edge_idx = utils.get_max_edge_idx(self.graph)
        self.node_gen = NodeGen(max_node_idx + 1, max_edge_idx + 1)
        self.name_edge_mapping = utils.name_obj_dict(self.graph.value_info)

    @staticmethod
    def make_value_info(name, tensor_type, shape):
        return onnx.helper.make_tensor_value_info(name, tensor_type, shape)

    def set_ins_place(self, ins_idx: int):
        self.ins_idx = ins_idx

    def set_ins_place_by_node_name(self, node_name):
        nodes_name = [node.name for node in self.graph.node]
        self.ins_idx = nodes_name.index(node_name) + 1

    def get_node_idx_by_output(self, edge):
        return [i for i, n in enumerate(self.graph.node)
                if edge.name in n.output][0]

    def get_node_output_by_node_idx(self, idx):
        node = self.get_node(idx)
        return self.name_edge_mapping[node.output[0]]

    def ins_slice(self, tgt_shape, in_edge, broadcast=False):
        in_shape = shape_utils.get_dim(in_edge)
        slice_shape = shape_utils.get_slice_shape(
            in_shape, tgt_shape,
            broadcast)
        if slice_shape != in_shape:
            op_type = 'Slice'
            slice_dict = attr_gen.slice_node(in_shape, slice_shape)
            # slice_output = self.ins_node_with_tensor(op_type, slice_dict, in_edge, slice_shape)
            slice_output = self.ins_node_with_attr(op_type, slice_dict, in_edge, slice_shape)
        else:
            slice_output = in_edge
        return slice_output

    def ins_unsqueeze(self, tgt_shape, in_edge):
        in_shape = shape_utils.get_dim(in_edge)
        if len(in_shape) < len(tgt_shape):
            axes, unsqueeze_shape = \
                attr_gen.unsqueeze_node(in_shape, len(tgt_shape))
            unsqueeze_output = self.ins_node_with_attr(
                'Unsqueeze', axes, in_edge, unsqueeze_shape)
        else:
            unsqueeze_output = in_edge
        return unsqueeze_output

    def ins_reduce(self, in_edge, reduce='mean', keep_dims=False, rank=2):
        assert reduce.lower() in ['mean', 'max', 'min', 'L1', 'L2', 'sum']
        op_type = "Reduce%s%s" % (reduce[0].upper(), reduce[1:])
        in_shape = shape_utils.get_dim(in_edge)
        if len(in_shape) <= rank:
            return in_edge
        else:
            attr, reduce_shape = attr_gen.reduce_node(in_shape, keep_dims, rank)
            reduce_output = self.ins_node_with_attr(
                op_type, attr, in_edge, reduce_shape
            )
            return reduce_output

    def ins_pad(self, in_edge, tgt_shape, broadcast, mode='constant'):
        src_shape = shape_utils.get_dim(in_edge)
        pad_shape = shape_utils.get_pad_shape(src_shape, tgt_shape, broadcast)
        if pad_shape != src_shape:
            pad_dict = attr_gen.pad_node(src_shape, pad_shape, mode)
            pad_output = self.ins_node_with_attr(
                'Pad', pad_dict, in_edge, pad_shape)
        else:
            pad_output = in_edge
        return pad_output

    def ins_constant(self, val):
        attr_dict = {'value': self.node_gen.new_tensor(
            val, self.node_gen.new_node_name('Constant'), 'tensor'
        )}
        return self.ins_node_with_attr('Constant', attr_dict, [], tuple(val.shape))

    def ins_node_with_attr(self, op_type, attr_dict, in_edges, output_shape):
        in_edges = utils.convert2iter(in_edges)
        node = self.node_gen.new_node(op_type,
                                      *[e.name for e in in_edges],
                                      **attr_dict)
        if not in_edges:
            out_type = onnx.TensorProto.FLOAT
        else:
            out_type = shape_utils.get_type(in_edges[0])
        output_edge = self.ins_edge_node(out_type, node, output_shape)
        return output_edge

    def ins_node_with_tensor(self, op_type, attr_dict, in_edges, output_shape):
        in_edges = utils.convert2iter(in_edges)
        node, attr_tensors = self.node_gen.new_node_with_tensor(
            op_type, attr_dict, *[e.name for e in in_edges]
        )
        if not in_edges:
            out_type = onnx.TensorProto.FLOAT
        else:
            out_type = shape_utils.get_type(in_edges[0])
        output_edge = self.ins_edge_node(out_type, node, output_shape)
        for t in attr_tensors:
            self.graph.initializer.append(t)
        return output_edge

    def ins_edge_node(self, out_type, onnx_node, output_shape):
        output_edge = self.make_value_info(
            onnx_node.output[0], out_type,
            output_shape
        )
        self.ins_nodes(onnx_node)
        self.ins_edges(output_edge)
        return output_edge

    def ins_node_class_edge(self, node, output_shape):
        onnx_node = self.node_gen.new_node_from_node_class(node)
        out_edge = self.ins_edge_node(
            shape_utils.get_type(node.get_all_edges()[0]),
            onnx_node,
            output_shape)
        return out_edge

    def ins_node_class_with_specified_output(self, node, output_edges):
        output_edges = utils.convert2iter(output_edges)
        onnx_node = self.node_gen.new_node_with_specified_output(
            node, output_edges
        )
        self.ins_nodes(onnx_node)
        self.ins_edges(output_edges)

    def ins_nodes(self, onnx_nodes):
        onnx_nodes = utils.convert2iter(onnx_nodes)
        for item in onnx_nodes:
            self.graph.node.insert(self.ins_idx, item)
            self.ins_idx += 1

    def ins_edges(self, edges):
        edges = utils.convert2iter(edges)
        for e in edges:
            self.graph.value_info.append(e)
        self.update_edges(edges)

    def update_edges(self, edges):
        # self.edges = utils.get_ordered_inner_edges(self.graph)
        edges = utils.convert2iter(edges)
        self.name_edge_mapping.update({e.name: e for e in edges})

    def gen_ins_node(self, op_type, in_edges_name):
        node = Node(op_type, len(in_edges_name))
        in_edges = [self.name_edge_mapping[e] for e in in_edges_name]
        node.in_edges.extend(in_edges)
        return node

    def add_output(self, output_edge):
        self.graph.output.append(output_edge)

    def make_edge(self, elem_type, tensor_shape):
        edge = self.node_gen.new_edge(elem_type, tensor_shape)
        self.ins_edges(edge)
        return edge

    def get_node(self, node_idx):
        return self.graph.node[node_idx]

    @staticmethod
    def replace_node_output(node, ori_edge, new_edge):
        node.output.remove(ori_edge.name)
        node.output.insert(0, new_edge.name)

    def get_edge_by_name(self, edge_name):
        return self.name_edge_mapping[edge_name]

    def get_inner_valid_nodes(self):
        # n_v = utils.name_obj_dict(self.graph.value_info)
        output_names = [e.name for e in self.graph.output]
        nodes = [node.name for node in self.graph.node
                 if not node.output[0] in output_names and
                 shape_utils.is_float32(self.name_edge_mapping[node.output[0]])]
        return nodes

    def get_node_idx_by_name(self, node_name):
        return [node.name for node in self.graph.node].index(node_name)

    def get_node_output_by_node_name(self, node_name):
        node_idx = self.get_node_idx_by_name(node_name)
        node = self.graph.node[node_idx]
        return self.name_edge_mapping[node.output[0]]


class GraphMutator:
    def __init__(self, model):
        self.graph_gen = GraphGen(model)
        self.tmp_path = "/export/d1/dwxiao/TVM/tmp_models/tmp.onnx"

    def print_model(self):
        print(self.graph_gen.model)

    def print_graph(self):
        print(onnx.helper.printable_graph(self.graph_gen.graph))

    def check_model(self):
        onnx.checker.check_model(self.graph_gen.model)

    def get_model(self):
        return self.graph_gen.model

    def get_edge_by_name(self, edge_name):
        return self.graph_gen.get_edge_by_name(edge_name)

    def add_output(self, edge):
        return self.graph_gen.add_output(edge)

    def gen_ins_node(self, op_type, input_names=None):
        return self.graph_gen.gen_ins_node(op_type, input_names)

    def set_ins_place(self, ins_idx):
        self.graph_gen.set_ins_place(ins_idx)

    def get_num_nodes(self):
        return len(self.graph_gen.graph.node)

    def ins_guard(self, guard_edge, input_data):
        ins_idx = self.graph_gen.get_node_idx_by_output(guard_edge) + 1
        self.graph_gen.set_ins_place(ins_idx)
        reduce_out = self.graph_gen.ins_reduce(guard_edge, keep_dims=True)
        guard_val = utils.get_internal_edge_output(self.get_model(), reduce_out,
                                                   input_data, self.tmp_path)
        guard_val_edge = self.graph_gen.ins_constant(guard_val)
        cmp_node = Node('Sub', 2, [reduce_out, guard_val_edge])
        return self.graph_gen.ins_node_class_edge(cmp_node, tuple(guard_val.shape))

    def ins_fcb(self, guard_node_name, live_node_name, dead_edge, input_data):
        guard_edge = self.get_node_output_by_node_name(guard_node_name)
        guard_edge = self.ins_guard(guard_edge, input_data)

        live_idx = self.graph_gen.get_node_idx_by_name(live_node_name)
        # live_idx = self.graph_gen.get_node_idx_by_output(live_edge)

        self.graph_gen.set_ins_place(live_idx + 1)

        live_node = self.graph_gen.get_node(live_idx)
        ori_live_edge = self.graph_gen.get_node_output_by_node_idx(live_idx)
        cond_node = Node('Mul', 2, [guard_edge, dead_edge])
        cond_out = self.bilateral_shape_matching(cond_node)
        cond_matched_out = self.unilateral_shape_matching(
            cond_out, shape_utils.get_dim(ori_live_edge), True
        )
        # TODO: replace live_edge RIGHT NOW!
        new_live_edge = self.graph_gen.make_edge(
            np.float32, shape_utils.get_dim(ori_live_edge))
        add_node = Node('Add', 2, [cond_matched_out, new_live_edge])
        self.graph_gen.ins_node_class_with_specified_output(add_node, ori_live_edge)
        self.graph_gen.replace_node_output(live_node, ori_live_edge, new_live_edge)

    def bilateral_shape_matching(self, ins_node):
        """
        Insert expand and slice nodes so as to match all the input edges' shape
        :type ins_node: Node
        """
        shapes = [shape_utils.get_dim(e) for e in ins_node.get_all_edges()]
        common_shape = shape_utils.get_common_shape(shapes, ins_node.broadcast())
        for i in range(0, len(shapes)):
            e = ins_node.get_edge(i)
            unsqueeze_output = self.graph_gen.ins_unsqueeze(common_shape, e)
            slice_output = self.graph_gen.ins_slice(common_shape, unsqueeze_output,
                                                    ins_node.broadcast())
            ins_node.set_edge(i, slice_output)
        return self.graph_gen.ins_node_class_edge(ins_node, common_shape)

    def unilateral_shape_matching(self, in_edge, tgt_shape, broadcast):
        # TODO: substitute in_edge of Add with the shape matching output edge
        src_shape = shape_utils.get_dim(in_edge)
        if len(src_shape) < len(tgt_shape):
            rank_out = self.graph_gen.ins_unsqueeze(tgt_shape, in_edge)
        elif len(src_shape) > len(tgt_shape):
            rank_out = self.graph_gen.ins_reduce(in_edge, 'max', rank=len(tgt_shape))
        else:
            rank_out = in_edge
        slice_out = self.graph_gen.ins_slice(tgt_shape, rank_out, False)
        pad_out = self.graph_gen.ins_pad(slice_out, tgt_shape, broadcast)
        return pad_out

    def get_node_output_by_node_idx(self, idx):
        return self.graph_gen.get_node_output_by_node_idx(idx)[0]

    def get_inner_valid_nodes(self):
        return self.graph_gen.get_inner_valid_nodes()

    def get_node_output_by_node_name(self, node_name):
        return self.graph_gen.get_node_output_by_node_name(node_name)

    def set_ins_place_by_node_name(self, node_name):
        return self.graph_gen.set_ins_place_by_node_name(node_name)


class RandomGraphMutator:
    def __init__(self, model):
        self.mut = GraphMutator(model)

    def mutate(self, input_data):
        nodes = self.mut.get_inner_valid_nodes()
        ins_end = random.randint(4, len(nodes) - 1)
        live_node = nodes[ins_end]

        guard_idx = random.randint(0, ins_end - 1)
        guard_node = nodes[guard_idx]

        dead_node_class = self.rand_gen_ins_nodes(1)[0]
        dead_node_name = self.rand_set_ins_place_edges(
            dead_node_class, nodes[:ins_end])

        self.mut.set_ins_place_by_node_name(dead_node_name)
        dead_edge = self.mut.bilateral_shape_matching(dead_node_class)

        self.mut.ins_fcb(guard_node, live_node, dead_edge, input_data)

    def rand_set_ins_place_edges(self, ins_node: Node, node_list):
        ids = []
        for i in range(0, ins_node.num_in + 1):
            ids.append(random.randint(0, len(node_list) - 1))
        ids.sort()
        ins_place_node = node_list[ids[-1]]
        for i in range(0, len(ids) - 1):
            # TODO: get node object; set in_edges to be its output
            # TODO: whether store node object or node name
            edge_node = node_list[ids[i]]
            in_edge = self.mut.get_node_output_by_node_name(edge_node)
            ins_node.add_edge(in_edge)
        return ins_place_node

    def rand_get_edge(self, end_idx, st_idx=0, return_idx=False):
        for i in range(0, 10):
            idx = random.randint(st_idx, end_idx)
            edge = self.mut.get_node_output_by_node_idx(idx)
            if shape_utils.is_float32(edge):
                if return_idx:
                    return idx
                return edge
        raise Exception("Cannot randomly find an edge with a floating type")

    def rand_gen_ins_end(self):
        """
        At least two internal edges including the end node's output
        :return: The index of node whose output is used to be
        added with the inserted model's output
        """
        return self.rand_get_edge(self.mut.get_num_nodes() - 2,
                                  st_idx=4, return_idx=True)

    @staticmethod
    def rand_gen_ins_nodes(num_ins):
        nodes = []
        for i in range(0, num_ins):
            # nodes.append(Node('Add', 2))
            if random.random() < 0.5:
                nodes.append(Node('Add', 2))
            else:
                nodes.append(Node('Sub', 2))
        return nodes
