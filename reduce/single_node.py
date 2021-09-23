import numpy as np
import onnx

from mutation.node_gen import NodeChainGen
from mutation.edge_node import convert_edge_to_value_info

g = NodeChainGen(1, 1)
a = g.make_constant(np.zeros((1, 1), dtype=np.float32))
b = g.make_constant(np.zeros((2, 2), dtype=np.float32))
c = g.make_constant(np.zeros((2, 2), dtype=np.float32))
out = g.make_edge_node('Sum', [a, b, c], (2, 2), False)

onnx_out = convert_edge_to_value_info(out)

graph_def = onnx.helper.make_graph(
 [a.def_node, b.def_node, c.def_node, out.def_node],
 'graph',
 [],
 [onnx_out]
)

model_def = onnx.helper.make_model(graph_def, producer_name='glow-error')
