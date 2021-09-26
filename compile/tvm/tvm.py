import os
import time

import numpy as np
import onnx
import tvm
from tvm import TVMError

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
        tvm_build.write_in_out_info(os.path.join(build_dir, "in_out.txt"), onnx_model)

    def run(self, run_dir):
        gmod = tvm_build.load_lib(run_dir)
        has_input, has_two_output = tvm_build.read_in_out_info(
            os.path.join(run_dir, "in_out.txt"))
        result = tvm_build.run_graph_module(gmod, has_input, has_two_output, self.input_data)
        start_time = time.time()
        if self.cal_time:
            for _ in range(5):
                tvm_build.run_graph_module(gmod, has_input, has_two_output, self.input_data)
        end_time = time.time()
        if has_two_output:
            np.save(os.path.join(run_dir, "out.npy"), result[0])
            np.save(os.path.join(run_dir, "edge.npy"), result[1])
        else:
            np.save(os.path.join(run_dir, "out.npy"), result)
        return end_time - start_time

    @staticmethod
    def get_output(run_dir):
        return np.load(os.path.join(run_dir, 'out.npy'))

    @staticmethod
    def get_edge_value(run_dir):
        return np.load(os.path.join(run_dir, "edge.npy"))
