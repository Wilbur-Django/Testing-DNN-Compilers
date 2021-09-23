import os.path

import onnx
from onnx import shape_inference

from mutation.make_module import make_relu, extend_graph
from mutation import utils
from shape_utils import get_dim


# class FCBMutation:
#     def __init__(self, seed_model):
#         self.seed_model = shape_inference.infer_shapes(seed_model)
#         onnx.checker.check_model(self.seed_model)
#         print('Inferred model is checked!')

def mutate(seed_model, mutate_times, save_dir):
    for i in range(1, mutate_times + 1):
        mutated_model, ins_start, ins_end = fcb_mutation(seed_model)
        save_path = os.path.join(save_dir, "%d_s%s_e%s.onnx" % (i, ins_start, ins_end))
        onnx.save(mutated_model, save_path)
        seed_model = mutated_model


def fcb_mutation(seed_model):
    model = shape_inference.infer_shapes(seed_model)
    del seed_model
    onnx.checker.check_model(model)
    print('Inferred model is checked!')

    # print(onnx.helper.printable_graph(inferred_model.graph))
    # print(inferred_model.graph.input)
    # print(inferred_model.graph.value_info)
    value_name_list = utils.get_value_name_list(model.graph)
    node_names = [t.name for t in model.graph.node]
    # print(node_names)

    next_edge_idx = utils.get_max_name_idx(value_name_list) + 1
    next_node_idx = utils.get_max_name_idx(node_names) + 1

    input_name, output_name = utils.get_ins_start_end(model.graph, lambda x, y: get_dim(x) == get_dim(y))

    # next_node_idx, next_edge_idx = 100, 100
    ins_node_list, next_node_idx, next_edge_idx = make_relu(input_name, next_node_idx, next_edge_idx)
    next_node_idx, next_edge_idx = extend_graph(model.graph, output_name, ins_node_list, next_node_idx, next_edge_idx)
    onnx.checker.check_model(model)
    print('New model is checked!')
    # print(onnx.helper.printable_graph(model.graph))
    return model, input_name, output_name
