from onnx import helper


def make_relu(input_name, next_node_idx, next_edge_idx):
    relu1 = helper.make_node(
        "Relu",
        [input_name],
        ["%d" % next_edge_idx],
        "Relu_%d" % next_node_idx
    )
    next_node_idx += 1
    next_edge_idx += 1
    neg = helper.make_node(
        "Neg",
        ["%d" % (next_edge_idx - 1)],
        ["%d" % next_edge_idx],
        "Neg_%d" % next_node_idx
    )
    next_node_idx += 1
    next_edge_idx += 1
    relu2 = helper.make_node(
        "Relu",
        ["%d" % (next_edge_idx - 1)],
        ["%d" % next_edge_idx],
        "Relu_%d" % next_node_idx
    )
    next_node_idx += 1
    next_edge_idx += 1
    return [relu1, neg, relu2], next_node_idx, next_edge_idx


def extend_graph(graph, output_name, ins_list, next_node_idx, next_edge_idx):
    node_list = graph.node
    output_node_idx, output_node = [(i, t) for i, t in enumerate(node_list)
                                    if output_name in t.output][0]
    ori_out = output_node.output
    ori_out.remove(output_name)
    add_in1 = "%d" % next_edge_idx
    ori_out.append(add_in1)
    next_edge_idx += 1
    add_in2 = ins_list[-1].output[0]
    add_node = helper.make_node(
        "Add",
        [add_in1, add_in2],
        [output_name],
        "Add_%d" % next_node_idx
    )
    next_node_idx += 1
    ins_list.append(add_node)

    ins_idx = output_node_idx + 1
    for i, m in enumerate(ins_list):
        node_list.insert(ins_idx + i, m)
    return next_node_idx, next_edge_idx
