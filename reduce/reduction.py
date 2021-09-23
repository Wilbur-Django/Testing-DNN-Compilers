from mutation import utils


def get_in_out_graph(onnx_graph):
    non_node_def_edges = utils.non_node_output_edges(onnx_graph)
    edges_in_out = {e: (node.name, []) for node in onnx_graph.node for e in node.output}
    for node in onnx_graph.node:
        for in_edge in node.input:
            if in_edge not in non_node_def_edges:
                edges_in_out[in_edge][1].append(node.name)
    out_node_mapping = {v[0]: v[1] for v in edges_in_out.values()}
    in_node_mapping = {node.name: [] for node in onnx_graph.node}
    for k, v in out_node_mapping.items():
        for out_node in v:
            in_node_mapping[out_node].append(k)
    return out_node_mapping, in_node_mapping


def propagate(cur_node, out_mapping, in_mapping, keep_dict, forward=False):
    # if cur_node == 'Conv_13' or cur_node == 'Relu_14':
    #     print(forward)
    #     print(keep_dict)
    if forward:
        for node in out_mapping[cur_node]:
            if keep_dict[node]:
                continue
            else:
                keep_dict[node] = True
                propagate(node, out_mapping, in_mapping, keep_dict)
                propagate(node, out_mapping, in_mapping, keep_dict, True)
    else:
        for node in in_mapping[cur_node]:
            if keep_dict[node]:
                continue
            else:
                keep_dict[node] = True
                propagate(node, out_mapping, in_mapping, keep_dict)


def reconnect_nodes(ori_graph, mut_graph):
    name_node_mapping = utils.name_obj_dict(mut_graph.node)
    non_node_def_edges = utils.non_node_output_edges(mut_graph)
    node_def_edges = set(e for node in mut_graph.node for e in node.output)
    ori_edges_def = {e: node for node in ori_graph.node for e in node.output}
    for node in mut_graph.node:
        for in_edge in node.input:
            if in_edge not in non_node_def_edges and in_edge not in node_def_edges:
                edge_def_node_name = ori_edges_def[in_edge].name
                edge_def_node = name_node_mapping[edge_def_node_name]
                # Assume there's only one output for every node
                edge_def_node.output.remove(edge_def_node.output[0])
                edge_def_node.output.insert(0, in_edge)


def get_reduction_set(ori_graph, mut_graph, keep_node_name):
    ori_nodes_name = set(node.name for node in ori_graph.node)
    keep_dict = {node.name: (node.name in ori_nodes_name) for node in mut_graph.node}
    keep_dict[keep_node_name] = True
    out_mapping, in_mapping = get_in_out_graph(mut_graph)
    propagate(keep_node_name, out_mapping, in_mapping, keep_dict)
    propagate(keep_node_name, out_mapping, in_mapping, keep_dict, forward=True)
    del_nodes_name = [k for k, v in keep_dict.items() if not v]
    print(len(del_nodes_name))
    name_node_mapping = utils.name_obj_dict(mut_graph.node)
    del_nodes = [name_node_mapping[name] for name in del_nodes_name]
    for node in del_nodes:
        mut_graph.node.remove(node)
    reconnect_nodes(ori_graph, mut_graph)
