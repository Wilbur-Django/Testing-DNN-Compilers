import datetime
import os
import random
import re

import onnx
import onnxruntime as rt

import numpy as np

from mutation.shape_utils import shape_match, get_dim


def name_obj_dict(objs):
    return {obj.name: obj for obj in objs}


def convert2iter(o):
    if not isinstance(o, tuple) and not isinstance(o, list):
        return [o]
    else:
        return o


def numpy_onnx_type_mapping(np_type):
    if np_type == np.float32:
        return onnx.TensorProto.FLOAT
    elif np_type == np.int32:
        return onnx.TensorProto.INT32
    elif np_type == np.int64:
        return onnx.TensorProto.INT64
    else:
        raise Exception("The type cannot be matched to onnx type")


def get_constant_edge_val(model, edge):
    node = [n for n in model.graph.node if edge.name in n.output]
    if not node:
        return
    node = node[0]
    if node.op_type != 'Constant':
        return
    val = node.attribute[0].t.float_data
    val_shape = node.attribute[0].t.dims
    val = np.array(list(val), dtype=np.float32).reshape(val_shape)
    return val


def get_internal_edge_output(model, edge, input_data, temp_save_path):
    val = get_constant_edge_val(model, edge)
    if val is not None:
        return val
    model.graph.output.insert(0, edge)
    onnx.save(model, temp_save_path)

    out_list = onnx_run(input_data, temp_save_path)
    model.graph.output.remove(edge)
    return out_list[0]


def onnx_run(input_data, model_path):
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = [o.name for o in sess.get_outputs()]
    out = sess.run(output_name, {input_name: input_data})
    return out


def print_onnx_graph(model):
    print(onnx.helper.printable_graph(model.graph))


def get_max_name_idx(name_list):
    pattern = re.compile(r"\d+")
    max_idx = 0
    for name in name_list:
        m = pattern.findall(name)
        if not m:
            continue
        max_idx = max(max([int(t) for t in m]), max_idx)
    return max_idx


def get_max_node_idx(graph):
    return get_max_name_idx([n.name for n in graph.node])


def get_max_edge_idx(graph):
    input_names = [i for n in graph.node for i in n.input]
    output_names = [o for n in graph.node for o in n.output]
    input_names.extend(output_names)
    return get_max_name_idx(input_names)


def get_ordered_inner_edges(graph):
    value_info_name_mapping = name_obj_dict(graph.value_info)
    edge_def_order = [out for node in graph.node for out in node.output]
    value_info_name = set(v.name for v in graph.value_info)
    inner_edges_name = list(set(edge_def_order).intersection(value_info_name))
    inner_edges_name.sort(key=edge_def_order.index)
    return [value_info_name_mapping[edge] for edge in inner_edges_name]


def get_value_name_list(graph):
    names = [t.name for t in graph.value_info]
    names.extend([t.name for t in graph.input])
    names.extend([t.name for t in graph.output])
    names.extend([t.name for t in graph.initializer])
    return names


def non_node_output_edges(graph):
    non_node_def_edges = set(e.name for e in graph.initializer)
    non_node_def_edges.update(set(e.name for e in graph.input))
    return non_node_def_edges


def get_ins_start_end(graph, shape_constrain):
    inner_edges = get_ordered_inner_edges(graph)
    for i in range(0, 5):
        start_idx = random.randrange(0, len(inner_edges))
        input_edge = inner_edges[start_idx]
        matched = shape_match(inner_edges, start_idx, input_edge, shape_constrain)
        if not matched:
            continue
        end_idx = random.randrange(0, len(matched))
        output_name = matched[end_idx]
        # print(len(matched))
        # print(matched)
        print("Insertion start:", input_edge.name)
        print("Insertion end:", output_name)
        return input_edge.name, output_name


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def path_append_timestamp(save_dir):
    sub_dir_name = datetime.datetime.now().strftime("%m%d_%H%M%S")
    save_path = os.path.join(os.path.curdir, save_dir, sub_dir_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def seed_mut_path(root_dir):
    mutation_save_path = path_append_timestamp(root_dir)
    print(os.path.abspath(mutation_save_path))
    seed_save_path = os.path.join(mutation_save_path, "seed.onnx")
    return seed_save_path, mutation_save_path


def array_diff(out, cmp_array, debug=False):
    diff = np.abs(out - cmp_array)
    max_abs_diff = np.max(diff)
    max_abs_idx = np.argmax(diff)
    rel_diff = (diff / (cmp_array + 1e-9))
    max_rel_diff = np.max(rel_diff)
    max_rel_idx = np.argmax(rel_diff)
    if debug:
        print(out)
        print(cmp_array)
        print(diff)
        print(max_abs_diff, max_abs_idx)
        print(max_rel_diff, max_rel_idx)
    return max_abs_diff, max_abs_idx, max_rel_diff, max_rel_idx


def make_value_info(name, shape, tensor_type=onnx.TensorProto.FLOAT):
    return onnx.helper.make_tensor_value_info(name, tensor_type, shape)


def insert_list(ins_obj, items: list, ins_index):
    for item in reversed(items):
        ins_obj.insert(ins_index, item)


def replace_node_output(node, new_output_name):
    ori_output = [o for o in node.output]
    for output in ori_output:
        node.output.remove(output)
    node.output.insert(0, new_output_name)


def is_val_zero(np_value):
    return np.max(np.abs(np_value)) < 1e-7
