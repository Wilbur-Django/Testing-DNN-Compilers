import os
import shutil
import time

import numpy as np
import tensorflow as tf

import onnx
from onnx_tf.backend import prepare

from compile.dispatch import Runner
from compile.compile_err import CompilationError


class XlaRunner(Runner):
    def __init__(self, compiler_path, data_path, mode, cal_time):
        super().__init__(compiler_path, data_path, mode, cal_time)
        self.input_data = None
        self.set_input(data_path)

        file_dir = os.path.dirname(__file__)
        self.build_graph_file = os.path.join(file_dir, "build_graph.txt")
        self.build_so_file = os.path.join(file_dir, "build_so.txt")
        self.run = self.run_with_time if cal_time else self.run_without_time

    def set_input(self, data_file):
        self.input_data = np.load(data_file)

    def compile(self, model_path, build_dir):
        onnx2tf(model_path, build_dir)
        shutil.copyfile(os.path.join(build_dir, "graph.pb"),
                        os.path.join(self.compiler_path, "graph.pb"))

        last_wd = os.getcwd()
        os.chdir(self.compiler_path)

        shutil.copyfile(self.build_graph_file, "BUILD")
        # TODO: only show error and warning
        r = os.system("bazel build @org_tensorflow//:graph")
        if r:
            raise CompilationError(model_path)

        shutil.copyfile(self.build_so_file, "BUILD")
        r = os.system("bazel build @org_tensorflow//:libmodel.so")
        if r:
            raise CompilationError(model_path)

        os.chdir(last_wd)

        shutil.copyfile(os.path.join(self.compiler_path, "bazel-bin", "libmodel.so"),
                        os.path.join(build_dir, "libmodel.so"))

    def run_with_time(self, run_dir):
        libmodel = get_libmodel(run_dir)
        output = predict(libmodel, self.input_data)
        np.save(os.path.join(run_dir, "out.npy"), output)

        start = time.time()
        # Repeat 5 times
        for i in range(5):
            predict(libmodel, self.input_data)
        end = time.time()
        return (end - start) * 200

    def run_without_time(self, run_dir):
        libmodel = get_libmodel(run_dir)
        output = predict(libmodel, self.input_data)
        np.save(os.path.join(run_dir, "out.npy"), output)

    @staticmethod
    def get_output(run_dir):
        return np.load(os.path.join(run_dir, "out.npy"))


def predict(libmodel, x):
    x = np.require(x, np.float32, ('c', 'a'))
    y = np.require(np.zeros((4, 10)), np.float32, ('c', 'a', 'w'))
    libmodel.run(x, y, x.size, y.size)
    return y


def get_libmodel(run_dir):
    libmodel = np.ctypeslib.load_library('libmodel', run_dir)
    libmodel.run.argtypes = [
        np.ctypeslib.ndpointer(np.float32, ndim=4, shape=(4, 3, 32, 32), flags=('c', 'a')),
        np.ctypeslib.ndpointer(np.float32, ndim=2, shape=(4, 10), flags=('c', 'a', 'w')),
        np.ctypeslib.ctypes.c_int,
        np.ctypeslib.ctypes.c_int]
    return libmodel


def onnx2tf(model_path, build_dir):
    onnx_model = onnx.load(model_path)

    saved_model_path = os.path.join(build_dir, "tf_model")
    os.makedirs(saved_model_path, exist_ok=True)

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(saved_model_path)


    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.compat.v1.saved_model.loader.load(sess, ['serve'], saved_model_path)
        tf.compat.v1.train.write_graph(sess.graph, '', os.path.join(build_dir, "graph.pb"), as_text=False)
