import os
import shutil
import time

import numpy as np
import tensorflow as tf

import onnx
from onnx_tf.backend import prepare

from compile.runner import Runner
from compile.compile_utils import execute_cmd, write_in_out_info
from compile.xla.xla_err import XlaError
from compile.xla.compile_manager import select_manager


class XlaRunner(Runner):
    def __init__(self, compiler_path, data_path, mode, cal_time):
        super().__init__(compiler_path, data_path, mode, cal_time)
        self.input_data = None
        self.set_input(data_path)

        file_dir = os.path.dirname(__file__)
        self.build_graph_file = os.path.join(file_dir, "build_graph.txt")
        self.build_so_file = os.path.join(file_dir, "build_so.txt")

    def set_input(self, data_file):
        self.input_data = np.load(data_file)

    def compile(self, model_path, build_dir):
        onnx_model = onnx.load(model_path)
        onnx2tf(onnx_model, self.compiler_path)

        write_in_out_info(os.path.join(build_dir, "in_out.txt"), onnx_model)
        manager = select_manager(self.mode, os.path.join(build_dir, "in_out.txt"))
        manager.before_compile(self.compiler_path)

        last_wd = os.getcwd()
        os.chdir(self.compiler_path)

        shutil.copyfile(self.build_graph_file, "BUILD")
        # TODO: only show error and warning
        r = execute_cmd("bazel", "build", "@org_tensorflow//:graph")
        if r:
            raise XlaError(model_path, r)

        shutil.copyfile(self.build_so_file, "BUILD")
        r = execute_cmd("bazel", "build", "@org_tensorflow//:libmodel.so")
        if r:
            raise XlaError(model_path, r)

        os.chdir(last_wd)

    def run(self, run_dir):
        lib = get_lib(os.path.join(self.compiler_path, "bazel-bin"))
        manager = select_manager(self.mode, os.path.join(run_dir, "in_out.txt"))
        output = manager.predict(lib, self.input_data)
        np.save(os.path.join(run_dir, "out.npy"), output)
        if self.cal_time:
            return self.cal_run_time(manager, lib)

    def cal_run_time(self, manager, lib):
        start = time.time()
        # Repeat 5 times
        for i in range(5):
            manager.predict(lib, self.input_data)
        end = time.time()
        return (end - start) * 200

    @staticmethod
    def get_output(run_dir):
        return np.load(os.path.join(run_dir, "out.npy"))

    @staticmethod
    def get_edge_value(run_dir):
        return np.load(os.path.join(run_dir, "edge.npy"))


def get_lib(compile_output_dir):
    lib = np.ctypeslib.load_library('libmodel', compile_output_dir)
    return lib


def onnx2tf(onnx_model, build_dir):
    saved_model_path = os.path.join(build_dir, "tf_model")
    os.makedirs(saved_model_path, exist_ok=True)

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(saved_model_path)


    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.compat.v1.saved_model.loader.load(sess, ['serve'], saved_model_path)
        tf.compat.v1.train.write_graph(sess.graph, '', os.path.join(build_dir, "graph.pb"), as_text=False)
