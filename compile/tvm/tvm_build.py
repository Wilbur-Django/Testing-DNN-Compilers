import os
import time

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from mutation.shape_utils import get_dim


def build_model(onnx_model, build_dir):
    shape_dict = {i.name: get_dim(i) for i in onnx_model.graph.input}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    target = tvm.target.Target("llvm", host="llvm")
    with tvm.transform.PassContext(opt_level=2):
        lib = relay.build(mod, target=target, params=params)
    lib.export_library(os.path.join(build_dir, "compiled_lib.so"))


def load_lib(build_dir):
    dev = tvm.cpu(0)
    lib: tvm.runtime.Module = tvm.runtime.load_module(os.path.join(build_dir, "compiled_lib.so"))
    # Call the library factory function for default and create
    # a new runtime.Module, wrap with graph module.
    gmod = graph_executor.GraphModule(lib["default"](dev))
    return gmod


def run_graph_module(gmod, has_input, has_two_output, input_data=None):
    if has_input:
        gmod.set_input("input", tvm.nd.array(input_data))
    gmod.run()
    tvm_output = gmod.get_output(0).numpy()
    if has_two_output:
        return tvm_output, gmod.get_output(1).numpy()
    else:
        return tvm_output

def cal_run_time(gmod, has_input, input_data=None, repeat_times=20):
    st_time = time.time()
    for _ in range(repeat_times):
        if has_input:
            gmod.set_input("input", tvm.nd.array(input_data))
        gmod.run()
    ed_time = time.time()
    return (ed_time - st_time) * 1000 / repeat_times
