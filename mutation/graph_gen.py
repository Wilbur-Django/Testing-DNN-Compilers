import onnx
from onnx import shape_inference
from mutation import utils, attr_gen, match_shape
import random
import numpy as np


class Node:
    def __init__(self, op_type, num_in):
        self.op_type = op_type
        self.num_in = num_in
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

    def broadcast_constrain(self):
        # TODO: Replace if-else
        if self.op_type == 'Add' or self.op_type == 'Sub':
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

    def new_node_with_attr(self, node_type, attr_dict, *input_edges, **kwargs):
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

    @staticmethod
    def new_tensor(np_val, node_name, attr_name):
        if np_val.dtype == np.float32:
            data_type = onnx.TensorProto.FLOAT
        elif np_val.dtype == np.int32:
            data_type = onnx.TensorProto.INT32
        elif np_val.dtype == np.int64:
            data_type = onnx.TensorProto.INT64

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
        value_name_list = utils.get_value_name_list(self.graph)
        node_names = [t.name for t in self.graph.node]
        next_edge_idx = utils.get_max_name_idx(value_name_list) + 1
        next_node_idx = utils.get_max_name_idx(node_names) + 1
        self.node_gen = NodeGen(next_node_idx, next_edge_idx)
        # TODO: update self.edges and self.name_edge_mapping after adding an edge
        self.edges = utils.get_ordered_inner_edges(self.graph)
        self.name_edge_mapping = utils.name_obj_dict(self.edges)
        self.ins_nodes = []

    @staticmethod
    def make_value_info(name, tensor_type, shape):
        return onnx.helper.make_tensor_value_info(name, tensor_type, shape)

    def shape_matching(self, ins_node):
        """
        Insert expand and slice nodes so as to match all the input edges' shape
        :type ins_node: Node
        """
        shapes = [match_shape.get_dim(e) for e in ins_node.get_all_edges()]
        common_shape = match_shape.get_common_shape(shapes)
        for i in range(0, len(shapes)):
            e = ins_node.get_edge(i)
            unsqueeze_output = self.ins_unsqueeze(common_shape, e)
            slice_output = self.ins_slice(common_shape, unsqueeze_output,
                                          ins_node.broadcast_constrain())
            ins_node.set_edge(i, slice_output)
        return common_shape

    def ins_unsqueeze(self, tgt_shape, in_edge):
        in_shape = match_shape.get_dim(in_edge)
        if len(in_shape) < len(tgt_shape):
            axes, unsqueeze_shape = \
                attr_gen.unsqueeze_node(in_shape, len(tgt_shape))
            unsqueeze_output = self.ins_node_with_attr(
                'Unsqueeze', axes, in_edge, unsqueeze_shape)
        else:
            unsqueeze_output = in_edge
        return unsqueeze_output

    def ins_node_with_attr(self, op_type, attr_dict, in_edges, output_shape):
        in_edges = utils.convert2iter(in_edges)
        node = self.node_gen.new_node(op_type,
                                      *[e.name for e in in_edges],
                                      **attr_dict)
        output_edge = self.ins_edge_node(
            match_shape.get_type(in_edges[0]), node, output_shape)
        return output_edge

    # TODO: check insert_node, ins_node_with_attr, ins_node_with_tensor for value_info
    def ins_edge_node(self, out_type, onnx_node, output_shape):
        output_edge = self.make_value_info(
            onnx_node.output[0], out_type,
            output_shape
        )
        self.insert_nodes(onnx_node)
        self.insert_edges(output_edge)
        self.update_edges()
        return output_edge

    def ins_slice(self, tgt_shape, in_edge, broadcast=False):
        in_shape = match_shape.get_dim(in_edge)
        slice_shape = match_shape.get_slice_shape(
            in_shape, tgt_shape,
            broadcast)
        if slice_shape != in_shape:
            op_type = 'Slice'
            slice_dict = attr_gen.slice_node(in_shape, tgt_shape)
            slice_output = self.ins_node_with_tensor(op_type, slice_dict, in_edge, slice_shape)
        else:
            slice_output = in_edge
        return slice_output

    def ins_node_with_tensor(self, op_type, attr_dict, in_edges, output_shape):
        in_edges = utils.convert2iter(in_edges)
        node, attr_tensors = self.node_gen.new_node_with_attr(
            op_type, attr_dict, *[e.name for e in in_edges]
        )
        output_edge = self.ins_edge_node(
            match_shape.get_type(in_edges[0]), node, output_shape)
        for t in attr_tensors:
            self.graph.initializer.append(t)
        return output_edge

    def insert_nodes(self, nodes):
        nodes = utils.convert2iter(nodes)
        for item in nodes:
            self.graph.node.insert(self.ins_idx, item)
            self.ins_idx += 1

    def insert_edges(self, edges):
        edges = utils.convert2iter(edges)
        for e in edges:
            self.graph.value_info.append(e)

    def update_edges(self):
        self.edges = utils.get_ordered_inner_edges(self.graph)
        self.name_edge_mapping = utils.name_obj_dict(self.edges)

    def gen_ins_node(self, op_type, in_edges_name):
        node = Node(op_type, len(in_edges_name))
        in_edges = [self.name_edge_mapping[e] for e in in_edges_name]
        node.in_edges.extend(in_edges)
        return node

    def post_process(self, ins_idx, ins_node):
        self.ins_idx = ins_idx
        input_common_shape = self.shape_matching(ins_node)
        onnx_node = self.node_gen.new_node_from_node_class(ins_node)
        out_edge = self.ins_edge_node(
            match_shape.get_type(ins_node.get_all_edges()[0]),
            onnx_node,
            input_common_shape)
        self.graph.output.append(out_edge)


class RandomGraphGen(GraphGen):
    @staticmethod
    def rand_ins_offsets(num_ins, ins_end):
        # insert before the second node so that there are at least two edges
        # insertion place can be right before the next node after the end node
        ins_offsets = random.choices(tuple(range(2, ins_end + 2)), k=num_ins)
        ins_offsets.sort()
        for i in range(1, num_ins):
            ins_offsets[i] = ins_offsets[i] - ins_offsets[i - 1]
        return ins_offsets

    def rand_set_in_edges(self, ins_idx, ins_node):
        edges_before = utils.get_edges_before_node(
            self.graph.node, self.edges, ins_idx)
        while not ins_node.is_full():
            in_edge_idx = random.randint(0, len(edges_before) - 1)
            in_edge = edges_before[in_edge_idx]
            ins_node.add_edge(in_edge)

    def rand_gen_ins_end(self):
        """
        At least two internal edges including the end node's output
        :return: The index of node whose output is used to be
        added with the inserted model's output
        """
        return random.randint(1, len(self.graph.node) - 2)

    def rand_gen_ins_nodes(self, num_ins):
        for i in range(0, num_ins):
            self.ins_nodes.append(Node('Add', 2))
            # if random.random() < 0.5:
            #     self.ins_nodes.append(Node('Add', 2))
            # else:
            #     self.ins_nodes.append(Node('Sub', 2))
