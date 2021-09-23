import re
import random

import onnx
from onnx import shape_inference

from make_module import make_relu, extend_graph


def name_obj_dict(objs):
    return {obj.name: obj for obj in objs}


def shape_match(node_list, start_idx, name_edge_dict, match_edge):
    matched = []
    for node in node_list[start_idx+1:]:
        for node_output_name in node.output:
            try:
                node_output = name_edge_dict[node_output_name]
            except KeyError as e:
                continue
            if match_edge.type.tensor_type.elem_type != \
                    node_output.type.tensor_type.elem_type:
                continue
            out_dims = node_output.type.tensor_type.shape.dim
            match_dim = match_edge.type.tensor_type.shape.dim
            if out_dims != match_dim:
                continue
            matched.append(node_output_name)
    return matched


def get_max_name_idx(name_list):
    pattern = re.compile(r"\d+")
    max_idx = 0
    for name in name_list:
        m = pattern.findall(name)
        if not m:
            continue
        max_idx = max(max([int(t) for t in m]), max_idx)
    return max_idx


def get_value_name_list(graph):
    names = [t.name for t in graph.value_info]
    names.extend([t.name for t in graph.input])
    names.extend([t.name for t in graph.output])
    names.extend([t.name for t in graph.initializer])
    return names


def fcb_mutation(onnx_model):
    model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(model)
    print('Inferred model is checked!')

    # print(onnx.helper.printable_graph(inferred_model.graph))
    # print(inferred_model.graph.input)
    # print(inferred_model.graph.value_info)
    # value_name_list = get_value_name_list(model.graph)
    # node_names = [t.name for t in model.graph.node]
    # print(node_names)

    # next_edge_idx = get_max_name_idx(value_name_list) + 1
    # next_node_idx = get_max_name_idx(node_names) + 1

    input_name, output_name = get_ins_start_end(model)

    next_node_idx, next_edge_idx = 100, 100
    ins_node_list, next_node_idx, next_edge_idx = make_relu(input_name, next_node_idx, next_edge_idx)
    next_node_idx, next_edge_idx = extend_graph(model.graph, output_name, ins_node_list, next_node_idx, next_edge_idx)
    onnx.checker.check_model(model)
    print('New model is checked!')
    # print(onnx.helper.printable_graph(model.graph))
    onnx.save(model, "./saved_models/super_resolution_mutated.onnx")


def get_ins_start_end(model):
    edges = model.graph.value_info
    name_edge_dict = name_obj_dict(model.graph.value_info)
    for i in range(0, 5):
        start_idx = random.randrange(0, len(edges))
        input_edge = edges[start_idx]
        matched = shape_match(model.graph.node, start_idx, name_edge_dict, input_edge)
        if not matched:
            continue
        end_idx = random.randrange(0, len(matched))
        output_name = matched[end_idx]
        # print(len(matched))
        # print(matched)
        print(input_edge.name)
        print(output_name)
        return input_edge.name, output_name
