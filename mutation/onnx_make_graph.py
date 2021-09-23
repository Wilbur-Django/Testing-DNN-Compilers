import onnx
from onnx import helper, shape_inference
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


# Create one input (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 2, 2])
pads = helper.make_tensor_value_info('pads', TensorProto.FLOAT, [1, 3, 4])

value = helper.make_tensor_value_info('value', AttributeProto.FLOAT, [1])

# Create one output (ValueInfoProto)
Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [2, 2, 2])

# Create a node (NodeProto) - This is based on Pad-11
pad_nod = helper.make_node(
    'Pad',  # node name
    ['X', 'pads', 'value'],  # inputs
    ['Y'],  # outputs
    mode='constant',  # attributes
)

perm_node = helper.make_node("Transpose", ['X'], ['perm_X'], perm=[1, 0, 2])

add_node = helper.make_node("Add", ['perm_X', 'X'], ['Z'])

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [pad_nod, perm_node, add_node],
    'test-model',
    [X, pads, value],
    [Z],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example')

# print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')
inferred_model = shape_inference.infer_shapes(model_def)
onnx.checker.check_model(inferred_model)
print('New model is checked!')
# print(inferred_model.graph.value_info)
print(onnx.helper.printable_graph(inferred_model.graph))
