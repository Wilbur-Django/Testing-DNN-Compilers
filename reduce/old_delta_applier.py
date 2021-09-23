import copy

from onnx import shape_inference

from reduce.delta_split import Delta
from reduce import reduce_utils
from mutation.shape_utils import get_dim
from mutation.node_gen import NodeChainGen
from reduce.edge_info import PlainEdge, SubsEdge, set_subs_place


def add_by_dep(zero_dict, dep_edge, cons_edges):
    if cons_edges:
        zero_dict.update({e: zero_dict[dep_edge] for e in cons_edges})


def iter_cons_zero_dict(zero_dict: dict, delta):
    add_by_dep(zero_dict, delta.left_dep_edge, delta.left_dead_edges)
    add_by_dep(zero_dict, delta.right_dep_edge, delta.right_dead_edges)
    add_by_dep(zero_dict, delta.guard_dep_edge_name,
               delta.guard_input_edges + [delta.guard_const_edge])
    add_by_dep(zero_dict, delta.subs_ori_edge_name, [delta.subs_new_edge_name])
    zero_dict.update({e: True for e in
                      delta.guard_out_edges + delta.mul_out_edges})
    if zero_dict[delta.left_dep_edge] \
            and (delta.right_dep_edge is None
                 or zero_dict[delta.right_dep_edge]):
        zero_dict.update({e: True for e in delta.dead_out_edges})
    else:
        zero_dict.update({e: False for e in delta.dead_out_edges})


def construct_zero_dict(ori_edges_name, delta_list):
    zero_dict = {e: False for e in ori_edges_name}
    for delta in delta_list:
        iter_cons_zero_dict(zero_dict, delta)
    return zero_dict


class DeltaApplier:
    def __init__(self, ori_model, delta_list, max_node_idx, max_edge_idx):
        ori_edges_name = [n.output[0] for n in ori_model.graph.node]
        zero_dict = construct_zero_dict(ori_edges_name, delta_list[1:])

        self.edges_info = {}
        inferred_model = shape_inference.infer_shapes(ori_model)
        self.construct_ori_edges([e for e in inferred_model.graph.value_info])
        for delta in delta_list[1:]:
            self.construct_delta_edges(zero_dict, delta)

        self.delta_list = delta_list
        self.ori_model = ori_model

        self.max_node_idx = max_node_idx
        self.max_edge_idx = max_edge_idx

    def valid_range(self):
        return range(1, len(self.delta_list))

    @staticmethod
    def get_def_node(edge_name, delta):
        for n in delta.nodes:
            if edge_name in n.output:
                return n

    @staticmethod
    def get_non_subs_edge(delta):
        return delta.edges[:-1]

    def construct_delta_edges(self, zero_dict, delta: Delta):
        subs_ori_edge = self.edges_info[delta.subs_ori_edge_name]

        for e in self.get_non_subs_edge(delta):
            def_node = self.get_def_node(e.name, delta)
            dep_edges = [self.edges_info[i] for i in def_node.input]
            edge_info = PlainEdge(
                e.name, get_dim(e), def_node, dep_edges, zero_dict[e.name]
            )
            self.edges_info.update({edge_info.name: edge_info})

        mul_out_edge = self.edges_info[delta.mul_out_edges[-1]]
        subs_new_edge = SubsEdge(
            delta.subs_new_edge_name, subs_ori_edge.shape,
            delta.nodes[-1], [mul_out_edge, subs_ori_edge],
            subs_ori_edge.is_zero()
        )
        self.edges_info.update({subs_new_edge.name: subs_new_edge})

    def construct_ori_edges(self, ori_onnx_edges):
        edges = [PlainEdge(onnx_e.name, get_dim(onnx_e), None, None, False, True)
                 for onnx_e in ori_onnx_edges]
        self.edges_info.update({e.name: e for e in edges})

    def set_ins_place(self, delta, graph_nodes, chain_gen):
        delta_subs_edge = self.edges_info[delta.subs_new_edge_name]
        set_subs_place(delta_subs_edge, delta.potential_ins_nodes,
                       graph_nodes, self.edges_info, chain_gen)
        return delta_subs_edge

    def apply_subs(self, delta, graph_nodes, chain_gen):
        subs_edge = self.set_ins_place(delta, graph_nodes, chain_gen)
        return subs_edge.apply()

    def apply_non_subs(self, delta: Delta):
        retained_nodes = []
        for e in self.get_non_subs_edge(delta):
            edge_info = self.edges_info[e.name]
            cur_nodes = edge_info.apply()
            retained_nodes.extend(cur_nodes)
        return retained_nodes

    def apply(self, delta_ids):
        for info in self.edges_info.values():
            info.reset()
        model = copy.copy(self.ori_model)
        graph_nodes = [n for n in model.graph.node]
        chain_gen = NodeChainGen(self.max_node_idx + 1, self.max_edge_idx + 1)
        for delta_idx in delta_ids:
            delta = self.delta_list[delta_idx]
            non_subs_nodes = self.apply_non_subs(delta)
            subs_nodes = self.apply_subs(delta, graph_nodes, chain_gen)
            graph_nodes.extend(non_subs_nodes)
            graph_nodes.extend(subs_nodes)

            # if subs_nodes[-1].output[0] != delta.subs_ori_edge_name:
            #     raise Exception("Not inserting in the same place,\n"
            #                     "delta idx: %d\n"
            #                     "original: %s\n"
            #                     "now: %s\n" % (delta_idx + 1,
            #                                    delta.subs_ori_edge_name,
            #                                    subs_nodes[-1].output[0]))

        reduce_utils.make_model(graph_nodes, model)
        return model
