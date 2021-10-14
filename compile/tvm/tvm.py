import os
import time

import numpy as np
import onnx
from tvm import TVMError

import compile.compile_utils
from compile.runner import Runner
from compile.tvm.tvm_err import TvmError
from compile.tvm import tvm_build


class TVMRunner(Runner):
    def __init__(self, compiler_path, data_path, mode, cal_time):
        super().__init__(compiler_path, data_path, mode, cal_time)
        self.input_data = np.load(data_path)

    def set_input(self, input_file):
        self.input_data = np.load(input_file)

    def compile(self, model_path, build_dir):
        onnx_model = onnx.load(model_path)
        try:
            tvm_build.build_model(onnx_model, build_dir)
        except TVMError as e:
            raise TvmError(model_path, str(e))
        compile.compile_utils.write_in_out_info(os.path.join(build_dir, "in_out.txt"), onnx_model)

    def run(self, run_dir):
        gmod = tvm_build.load_lib(run_dir)
        has_input, has_two_output = compile.compile_utils.read_in_out_info(
            os.path.join(run_dir, "in_out.txt"))
        result = tvm_build.run_graph_module(gmod, has_input, has_two_output, self.input_data)
        if has_two_output:
            np.save(os.path.join(run_dir, "edge.npy"), result[0])
            np.save(os.path.join(run_dir, "out.npy"), result[1])
        else:
            np.save(os.path.join(run_dir, "out.npy"), result)

        if self.cal_time:
            return tvm_build.cal_run_time(gmod, has_input, input_data=self.input_data)

    @staticmethod
    def get_output(run_dir):
        return np.load(os.path.join(run_dir, 'out.npy'))

    @staticmethod
    def get_edge_value(run_dir):
        return np.load(os.path.join(run_dir, "edge.npy"))
