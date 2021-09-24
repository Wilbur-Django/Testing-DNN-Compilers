import os

import numpy as np
import onnx
import tvm
from tvm import TVMError

from compile.runner import Runner
from compile.tvm.tvm_utils import build_lib, get_context_target, lib_run
from compile.tvm.tvm_err import TvmError


class TVMRunner(Runner):
    def __init__(self, compiler_path, data_path, mode, cal_time):
        super().__init__(compiler_path, data_path, mode, cal_time)
        self.input_data = np.load(data_path)
        self.ctx, self.tgt = get_context_target(gpu=False)
        self.run = self.run_with_time if cal_time else self.run_without_time

    def set_input(self, input_file):
        self.input_data = np.load(input_file)

    def compile(self, model_path, build_dir):
        model = onnx.load(model_path)
        try:
            lib = build_lib(model, self.input_data, self.tgt)
        except TVMError as e:
            raise TvmError(model_path, str(e))
        lib.export_library(os.path.join(build_dir, "lib.tar"))

    def run_with_time(self, run_dir):
        lib = tvm.runtime.load_module(os.path.join(run_dir, "lib.tar"))
        result, sec = lib_run(lib, self.input_data, self.ctx, True)
        np.save(os.path.join(run_dir, "out.npy"), result)
        return sec

    def run_without_time(self, run_dir):
        lib = tvm.runtime.load_module(os.path.join(run_dir, "lib.tar"))
        result = lib_run(lib, self.input_data, self.ctx, False)
        np.save(os.path.join(run_dir, "out.npy"), result)

    @staticmethod
    def get_output(run_dir):
        return np.load(os.path.join(run_dir, 'out.npy'))
