import copy
import os

import onnx

from mutation.edge_node import EdgeNode, convert_onnx_to_edge
from mutation.fcb_mut import read_mut_info
from mutation.node_gen import make_node_chain_generator
from mutation.utils import insert_list
from reduce.edge_info import PlainEdge, SubsEdge


def convert_edge_to_plain(edge: EdgeNode, name_info_mapping: dict):
    dep_edges_info = [name_info_mapping[e.name] for e in edge.def_node.input]
    edge_info = PlainEdge(edge.name, edge.shape, edge.def_node,
                          dep_edges_info, edge.zero)
    name_info_mapping.update({edge_info.name: edge_info})
    return edge_info


def convert_edge_to_subs(subs_ori_edge: EdgeNode, subs_new_edge: EdgeNode,
                         name_info_mapping: dict):
    add_node = subs_ori_edge.def_node
    add_in_name = [i for i in add_node.input if i != subs_new_edge.name][0]
    add_out_name = subs_ori_edge.name
    add_in_info = name_info_mapping[add_in_name]
    add_out_info = name_info_mapping[add_out_name]
    edge_info = SubsEdge(subs_new_edge.name, subs_new_edge.shape,
                         add_node, [add_in_info, add_out_info],
                         subs_new_edge.zero)
    name_info_mapping.update({edge_info.name: edge_info})
    return edge_info


def set_potential_ins_places(graph, subs_node, ori_node_name_set):
    pot_places = [subs_node.name]
    st_idx = [n.name for n in graph.node].index(pot_places[0]) + 2
    for node in graph.node[st_idx:]:
        pot_places.append(node.name)
        if node.name in ori_node_name_set:
            break
    return pot_places


def get_subs_place(graph_nodes, potential_ins_places):
    subs_edge_name = None
    subs_node = None
    idx = None
    graph_nodes_name = [n.name for n in graph_nodes]
    for potential_pos in potential_ins_places:
        try:
            idx = graph_nodes_name.index(potential_pos)
            subs_node = graph_nodes[idx]
            subs_edge_name = subs_node.output[0]
            break
        except ValueError:
            continue
    return idx, subs_edge_name, subs_node


def set_subs_place(delta_subs_edge, potential_ins_places,
                   graph_nodes, name_info_mapping, chain_gen):
    idx, subs_edge_name, subs_node = get_subs_place(
        graph_nodes, potential_ins_places)

    subs_edge_info = name_info_mapping[subs_edge_name]
    delta_subs_edge.set_subs(subs_edge_info, subs_node, chain_gen)
    return idx, subs_node


def construct_deltas(model_dir, edge_dir):
    seed_model = onnx.load(os.path.join(model_dir, "seed.onnx"))
    all_edges_info = convert_onnx_to_edge(seed_model.graph)
    name_info_mapping = {info.name: info for info in all_edges_info}
    ori_nodes_name_set = set(n.name for n in seed_model.graph.node)

    delta_list = []
    for file_name in os.listdir(model_dir):
        model_name = os.path.splitext(file_name)[0]
        if model_name == 'seed':
            continue
        model = onnx.load(os.path.join(model_dir, file_name))
        dead_edges, subs_new_edge, subs_ori_edge = read_mut_info(
            os.path.join(edge_dir, "%s.txt" % model_name)
        )
        delta = Delta(dead_edges, subs_ori_edge, subs_new_edge,
                      name_info_mapping, model.graph, ori_nodes_name_set)
        delta_list.append(delta)

    return seed_model, delta_list, name_info_mapping


class Delta:
    def __init__(self, dead_edges, subs_ori_edge,
                 subs_new_edge, name_info_mapping, graph, ori_nodes_name_set):
        self.non_subs_edges = [convert_edge_to_plain(e, name_info_mapping)
                               for e in dead_edges]
        self.subs_edge = convert_edge_to_subs(subs_ori_edge, subs_new_edge,
                                              name_info_mapping)
        self.potential_ins_places = set_potential_ins_places(
            graph, subs_new_edge.def_node, ori_nodes_name_set
        )

    def reset(self):
        for edge in self.non_subs_edges:
            edge.reset()
        self.subs_edge.reset()

    def apply(self, graph_nodes, name_info_mapping, generator):
        ins_idx, subs_node = set_subs_place(
            self.subs_edge, self.potential_ins_places,
            graph_nodes, name_info_mapping, generator)

        graph_nodes.remove(subs_node)

        new_nodes = []
        for edge in self.non_subs_edges:
            dead_nodes = edge.apply()
            new_nodes.extend(dead_nodes)

        subs_nodes = self.subs_edge.apply()
        subs_nodes.insert(-1, subs_node)

        new_nodes.extend(subs_nodes)

        insert_list(graph_nodes, new_nodes, ins_idx)


class DeltaApplier:
    def __init__(self, model_dir, edge_dir):
        self.seed_model, self.delta_list, self.name_info_mapping = \
            construct_deltas(model_dir, edge_dir)

    def reset(self):
        for delta in self.delta_list:
            delta.reset()

    def apply(self, delta_ids):
        model = copy.copy(self.seed_model)
        self.reset()
        gen = make_node_chain_generator(model)
        for delta_id in delta_ids:
            delta = self.delta_list[delta_id]
            delta.apply(model.graph.node, self.name_info_mapping, gen)

        return model
