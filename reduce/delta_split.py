import copy
import os
from collections import deque

import onnx

from mutation import utils
from reduce import reduce_utils


class Delta:
    def __init__(self):
        self.nodes = []
        self.edges = []

        self.st_node_idx = None
        self.end_node_idx = None

        self.st_edge_idx = None
        self.end_edge_idx = None

        self.subs_ori_edge_name = None
        self.subs_new_edge_name = None

        self.dead_nodes = None
        self.dead_dep_edges_name = None

        self.guard_nodes = None
        self.guard_dep_edge_name = None

        self.mul_nodes = None
        self.edges_after_mul = None
        self.potential_ins_nodes = None

        self.left_dep_edge = None
        self.right_dep_edge = None
        self.left_dead_edges = None
        self.right_dead_edges = None
        self.dead_out_edges = None

        self.guard_input_edges = None
        self.guard_const_edge = None
        self.guard_out_edges = None

        self.mul_out_edges = None

        self.dep_edges_name = None

    def partition_nodes(self):
        i = 0
        self.dead_nodes = []
        for node in self.nodes:
            self.dead_nodes.append(node)
            i += 1
            if node.name.startswith('Add') or node.name.startswith("Sub"):
                break
        adder_edges = self.dead_nodes[-1].input
        dead_edges = reduce_utils.get_edge_chain(
            adder_edges[0], self.dead_nodes, None, False)
        self.left_dep_edge = dead_edges[-1]
        if dead_edges[:-1]:
            self.left_dead_edges = dead_edges[:-1]

        if adder_edges[0] != adder_edges[1]:
            dead_edges = reduce_utils.get_edge_chain(
                adder_edges[1], self.dead_nodes, None, False
            )
            self.right_dep_edge = dead_edges[-1]
            if dead_edges[:-1]:
                self.right_dead_edges = dead_edges[:-1]

        self.guard_nodes = []
        for node in self.nodes[i:]:
            i += 1
            self.guard_nodes.append(node)
            if node.name.startswith('Sub'):
                break
        self.mul_nodes = [n for n in self.nodes[i:]]
        mul_node = [n for n in self.mul_nodes if "Mul" in n.name][0]

        guard_sub_node = self.guard_nodes[-1]
        guard_input_edges = reduce_utils.get_edge_chain(
            guard_sub_node.input[0], self.guard_nodes, None, False
        )
        if guard_input_edges[:-1]:
            self.guard_input_edges = guard_input_edges[:-1]
        self.guard_dep_edge_name = self.guard_input_edges[-1]
        self.guard_const_edge = guard_sub_node.input[1]
        self.guard_out_edges = reduce_utils.get_edge_chain(
            guard_sub_node.output[0], self.mul_nodes, mul_node.name, True
        )

        self.dead_out_edges = reduce_utils.get_edge_chain(
            self.dead_nodes[-1].output[0], self.mul_nodes, mul_node.name, True
        )

        subs_add_node = self.mul_nodes[-1]
        self.mul_out_edges = reduce_utils.get_edge_chain(
            mul_node.output[0], self.mul_nodes, subs_add_node.name, True
        )

    def construct(self, last_max_node_idx, graph, ori_graph):
        self.st_node_idx = last_max_node_idx + 1
        self.construct_nodes(graph)
        self.construct_edges(graph)
        self.partition_nodes()

        self.construct_subs_node()
        self.construct_ins_place(graph, ori_graph)
        self.construct_dep()

    def construct_nodes(self, graph):
        self.nodes = [n for n in graph.node
                      if reduce_utils.parse_node_idx(n) >= self.st_node_idx]
        self.nodes.sort(key=lambda x: reduce_utils.parse_node_idx(x))
        self.end_node_idx = reduce_utils.parse_node_idx(self.nodes[-1])

    def construct_edges(self, graph):
        min_edge_idx = int(self.nodes[0].output[0])
        max_edge_idx = max(int(o) for o in self.nodes[-1].input)
        self.st_edge_idx = min_edge_idx
        self.end_edge_idx = max_edge_idx
        self.edges = [e for e in graph.value_info
                      if int(e.name) in range(min_edge_idx, max_edge_idx + 1)]
        self.edges.sort(key=lambda x: int(x.name))

    def construct_subs_node(self):
        final_add_node = self.nodes[-1]
        self.subs_ori_edge_name = final_add_node.output[0]
        self.subs_new_edge_name = self.edges[-1].name

    def construct_ins_place(self, graph, ori_graph):
        ori_nodes = set(n.name for n in ori_graph.node)
        graph_names = [n.name for n in graph.node]
        subs_node = [n.name for n in graph.node
                     if n.output[0] == self.subs_new_edge_name][0]
        add_node_idx = graph_names.index(self.nodes[-1].name)
        self.potential_ins_nodes = [subs_node]
        for node_name in graph_names[add_node_idx + 1:]:
            self.potential_ins_nodes.append(node_name)
            if node_name in ori_nodes:
                break
        # self.edges_after_mul = self.get_edges_around(
        #     self.subs_ori_edge_name, graph_edges_list, ori_edges_set, False
        # )

    @staticmethod
    def get_edges_around(ins_name, graph_edges, ori_edges, before=False):
        cur_place = graph_edges.index(ins_name)
        edges_around = []
        if before:
            for e_name in reversed(graph_edges[0:cur_place]):
                edges_around.append(e_name)
                if e_name in ori_edges:
                    break
        else:
            for e_name in graph_edges[cur_place + 1:]:
                edges_around.append(e_name)
                if e_name in ori_edges:
                    break
        return edges_around

    def construct_dep(self):
        self.dep_edges_name = []

        dead_dep_edges_name = [i for n in self.dead_nodes for i in n.input
                               if int(i) not in range(self.st_edge_idx,
                                                      self.end_edge_idx + 1)]

        # self.dead_dep_nodes_name = [output_node_map[e].name
        #                             for e in dead_dep_edges_name]
        self.dead_dep_edges_name = list(set(dead_dep_edges_name))
        self.dep_edges_name.extend(self.dead_dep_edges_name)

        guard_dep_edge_name = str(min(int(i) for i in self.guard_nodes[0].input))
        self.guard_dep_edge_name = guard_dep_edge_name
        # self.guard_dep_node_name = output_node_map[guard_dep_edge_name].name

        self.dep_edges_name.extend([guard_dep_edge_name])

        # TODO: check whether nodes that reference subs_ori_edge_name should be
        #  added to dependent nodes
        # subs_dep_nodes_name = [output_node_map[self.subs_new_edge_name].name,
        #                        reduce_utils.find_node_name_by_edge(
        #                            graph, self.subs_ori_edge_name)]
        # TODO: check whether we should add the output of the first node that
        #  references the subs_ori_edge_name
        self.dep_edges_name.extend([self.subs_ori_edge_name])

        # TODO: whether removing self-referencing should be moved to before subs
        for dep_e in self.dep_edges_name:
            if self.is_edge_name_in_range(dep_e):
                self.dep_edges_name.remove(dep_e)

    def get_edges_name(self):
        return [e.name for e in self.edges]

    def get_nodes_name(self):
        return [n.name for n in self.nodes]

    def is_edge_name_in_range(self, edge_name):
        return int(edge_name) in range(self.st_edge_idx, self.end_edge_idx + 1)

    @staticmethod
    def insert_nodes(graph, dep_edges_name, ins_nodes):
        dep_edges_name = utils.convert2iter(dep_edges_name)
        # Assume one output per node
        graph_edges_name = [node.output[0] for node in graph.node]
        ins_idx = max(graph_edges_name.index(e) for e in dep_edges_name) + 1
        for node in ins_nodes:
            graph.node.insert(ins_idx, node)
            ins_idx += 1

    @staticmethod
    def flexible_insert_nodes(graph, dep_edges_name, ins_nodes):
        dep_edges_name = utils.convert2iter(dep_edges_name)
        # Assume one output per node
        graph_edges_name = [node.output[0] for node in graph.node]
        ins_ids = []
        for e in dep_edges_name:
            try:
                idx = graph_edges_name.index(e)
                ins_ids.append(idx)
            except ValueError:
                continue
        ins_idx = max(ins_ids) + 1
        for node in ins_nodes:
            graph.node.insert(ins_idx, node)
            ins_idx += 1

    def apply_to(self, graph):
        self.flexible_insert_nodes(
            graph,
            self.dead_dep_edges_name + self.edges_before_dead,
            self.dead_nodes
        )
        self.insert_nodes(graph, self.guard_dep_edge_name, self.guard_nodes)
        self.insert_nodes(graph, self.subs_ori_edge_name, self.mul_nodes)
        subs_node_idx = [node.output[0] for node in graph.node] \
            .index(self.subs_ori_edge_name)

        subs_node = graph.node[subs_node_idx]
        subs_node.output.remove(self.subs_ori_edge_name)
        subs_node.output.append(self.subs_new_edge_name)

        for edge in self.edges:
            graph.value_info.append(edge)


class DeltaSplitter:
    def __init__(self, test_end):
        self.root_dir = \
            "/export/d1/dwxiao/TVM/mutation/mutated_models/resnet18/0526_085234"
        self.base_model = onnx.load(os.path.join(self.root_dir, "seed.onnx"))
        self.delta_list = []
        self.dep_relation = []
        self.test_end = test_end
        self.construct_deltas()

    def construct_deltas(self):
        self.delta_list.append(None)
        last_max_node_id = utils.get_max_name_idx(
            [n.name for n in self.base_model.graph.node])
        for i in range(1, self.test_end + 1):
            model = onnx.load(os.path.join(self.root_dir, "%d.onnx" % i))
            d = Delta()
            d.construct(last_max_node_id, model.graph, self.base_model.graph)
            self.delta_list.append(d)
            last_max_node_id = d.end_node_idx

    def get_deltas(self):
        return self.delta_list[1:]

    def construct_dep_relation(self):
        self.dep_relation.append(None)
        for idx, delta in enumerate(self.get_deltas()):
            # dep = set(i for i in range(1, self.test_end + 1)
            #           for dep_n in delta.dep_nodes_name
            #           if dep_n in self.delta_list[i].get_nodes_name())
            dep = set(i for i in range(1, self.test_end + 1)
                      for dep_e in delta.dep_edges_name
                      if self.delta_list[i].is_edge_name_in_range(dep_e))
            dep = list(dep)
            if (idx + 1) in dep:
                dep.remove(idx + 1)
            dep.sort()
            self.dep_relation.append(tuple(dep))

    def get_dep_relation(self):
        return self.dep_relation[1:]

    def get_dep_chain(self, delta_ids):
        dep_chain = set()
        unprocessed = deque(delta_ids)
        while unprocessed:
            d = unprocessed.popleft()
            dep_chain.add(d)
            for dep in self.dep_relation[d]:
                if dep not in dep_chain:
                    unprocessed.append(dep)
        dep_chain = list(dep_chain)
        dep_chain.sort()
        return dep_chain

    def check_dep(self, delta_ids):
        dep_chain = self.get_dep_chain(delta_ids)
        if dep_chain != delta_ids:
            return False
        return True
        # for idx in dep_chain:
        #     if idx not in delta_ids:
        #         return False
        # for idx in delta_ids:
        #     dep_ids = self.dep_relation[idx]
        #     for dep_idx in dep_ids:
        #         if dep_idx not in delta_ids:
        #             return False
        # return True

    def apply_deltas(self, delta_ids):
        model = copy.copy(self.base_model)
        for idx in delta_ids:
            d = self.delta_list[idx]
            d.apply_to(model.graph)
            # TODO: move the checker out of the loop
        onnx.checker.check_model(model)
        return model
