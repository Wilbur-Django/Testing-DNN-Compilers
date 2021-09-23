import copy

import onnx
from onnx import shape_inference
import numpy as np
import os

from reduce import reduce_utils
from mutation.utils import name_obj_dict, onnx_run


class NodeApplier:
    def __init__(self, fault_model, delta_nodes, value_list):
        self.fault_model = fault_model
        self.delta_nodes = delta_nodes
        self.value_list = value_list

    def valid_range(self):
        return range(len(self.delta_nodes))

    def apply(self, delta_ids):
        model = copy.copy(self.fault_model)
        deleted_ids = set(range(len(self.delta_nodes))).difference(delta_ids)

        for idx in deleted_ids:
            node = self.delta_nodes[idx]
            value = self.value_list[idx]
            const_node = reduce_utils.make_constant(
                value, reduce_utils.parse_node_idx(node), int(node.output[0]))
            reduce_utils.replace_node(model.graph, node.name, const_node)

        reduce_utils.remove_unref_nodes(model.graph)
        onnx.checker.check_model(model)
        return model


def get_inner_non_const_nodes(model):
    return [n for n in model.graph.node
            if 'Constant' not in n.name and 'output' not in n.output]


def add_output(model, view_edges, model_save_dir):
    os.makedirs(model_save_dir, exist_ok=True)
    model = shape_inference.infer_shapes(model)

    name_edge_mapping = name_obj_dict(model.graph.value_info)
    all_edges = [name_edge_mapping[e_name] for e_name in view_edges]

    for edge_name, edge in zip(view_edges, all_edges):
        model.graph.output.insert(0, edge)
        onnx.save(model, os.path.join(model_save_dir, "%s.onnx" % edge_name))
        model.graph.output.remove(edge)


def get_edges_output(models_dir, output_dir, input_data):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(models_dir):
        model_name = os.path.splitext(file_name)[0]
        result_dir = os.path.join(output_dir, model_name)
        os.makedirs(result_dir, exist_ok=True)

        edge_out, out = onnx_run(input_data, os.path.join(models_dir, file_name))
        np.save(os.path.join(result_dir, "out.npy"), out)
        np.save(os.path.join(result_dir, "edge.npy"), edge_out)


def make_node_applier(reduced_model_path, input_file, model_save_dir, output_dir):
    model = onnx.load(reduced_model_path)
    delta_nodes = get_inner_non_const_nodes(model)
    view_edges = [n.output[0] for n in delta_nodes]

    add_output(model, view_edges, model_save_dir)
    get_edges_output(model_save_dir, output_dir, np.load(input_file))

    edge_val = [np.load(os.path.join(output_dir, n.output[0], "edge.npy"))
                for n in delta_nodes]

    return NodeApplier(model, delta_nodes, edge_val)
