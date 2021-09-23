import os

import onnx

from compile.compile_err import CompilationError

class OnnxRunner:
    def __init__(self, compiler_path):
        self.compiler_path = compiler_path
        self.input_file = None

    def set_input(self, input_file):
        self.input_file = input_file

    def compile(self, model_path, build_dir):
        model = onnx.load(model_path)
        edge = [n for n in model.graph.node if '390'in n.output]
        if not edge:
            return
        r = os.system(f"{self.compiler_path} compile/onnx_runner/insert_run.py {model_path} {self.input_file}"
                  f" {build_dir}")
        if r:
            print(r)
            raise CompilationError(model_path)
