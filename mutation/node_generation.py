import onnx
from mutation import utils
from match_shape import broadcast_constrain
import random


class Node:
    def __init__(self, op_type, num_in):
        self.op_type = op_type
        self.num_in = num_in
        self.in_edges = []

    def is_empty(self):
        return not self.in_edges

    def is_full(self):
        return self.num_in >= len(self.in_edges)

    def get_in_edge(self, i):
        return self.in_edges[i]

    def set_in_edge(self, e):
        self.in_edges.append(e)

    def is_shape_matched(self):
        assert self.num_in == len(self.in_edges)
        if self.op_type == 'Add':
            return broadcast_constrain(self.in_edges[0], self.in_edges[1])


class NodeGen:
    def __init__(self, next_node_idx, next_edge_idx):
        self.node_id = next_node_idx
        self.edge_id = next_edge_idx

    def new_node_name(self, node_type):
        return "%s_%d" % (node_type, self.node_id)

    def new_edge_name(self):
        return str(self.edge_id)

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


class RandomNodeGen:
    def __init__(self, graph):
        self.graph = graph
        value_name_list = utils.get_value_name_list(graph)
        node_names = [t.name for t in graph.node]
        next_edge_idx = utils.get_max_name_idx(value_name_list) + 1
        next_node_idx = utils.get_max_name_idx(node_names) + 1
        self.node_gen = NodeGen(next_node_idx, next_edge_idx)
        self.edges = utils.get_ordered_inner_edges(graph)
        self.ins_nodes = []

    def new_node(self, node_type, *args, **kwargs):
        return self.node_gen.new_node(node_type, *args, **kwargs)

    def single_in_single_out(self, node_type):
        pass

    def get_ins_end(self):
        """
        At least two internal edges including the end node's output
        :return: The index of node whose output is used to be added with the inserted model's output
        """
        return random.randint(1, len(self.graph.node))

    def gen_ins_nodes(self, num_ins):
        for i in range(0, num_ins):
            self.ins_nodes.append(Node('Add', 2))

    def gen_ins_places(self, num_ins):
        # insert after the 1st node so that there are at least two edges
        # insertion place can be at the end node
        ins_places = random.choices(tuple(range(1, self.get_ins_end() + 1)), k=num_ins)
        ins_places.sort()
        for i in range(1, num_ins):
            ins_places[i] = ins_places[i] - ins_places[i - 1]
        return ins_places

    def ins_one_node(self, ins_idx):
        ins_node = self.ins_nodes[0]
        edges_before = utils.get_edges_before_node(self.graph.node, self.edges, ins_idx)
        while not ins_node.is_full():
            in_edge_idx = random.randint(0, len(edges_before) - 1)
            in_edge = edges_before[in_edge_idx]
            ins_node.set_in_edge(in_edge)

