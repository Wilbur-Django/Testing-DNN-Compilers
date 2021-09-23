import onnx
import os
import torch

from utils import get_internal_out_list, get_ordered_inner_edges
from compile_utils import pytorch2onnx

from models import super_resolution

torch_model = super_resolution.get_model()
onnx_save_path = "/export/d1/dwxiao/TVM/tmp_models/super_resolution.onnx"
temp_save_path = "/export/d1/dwxiao/TVM/tmp_models/super_res_inner.onnx"
input_tensor = torch.randn(1, 1, 224, 224)
onnx_model = pytorch2onnx(torch_model, input_tensor, None,
                          onnx_save_path)
edge_list = get_ordered_inner_edges(onnx_model.graph)
edge = edge_list[1]

out = get_internal_out_list(onnx_model, edge, input_tensor.numpy(), temp_save_path)
