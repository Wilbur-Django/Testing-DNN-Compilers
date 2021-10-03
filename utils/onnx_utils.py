import re

import onnx
import onnxruntime as rt


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


def onnx_run(input_data, model_path):
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = [o.name for o in sess.get_outputs()]
    out = sess.run(output_name, {input_name: input_data})
    return out


def print_onnx_graph(model):
    print(onnx.helper.printable_graph(model.graph))


def name_obj_dict(objs):
    return {obj.name: obj for obj in objs}