import os
import shutil

from compile.compile_err import CompilationError
from compile.glow.form_cpp import form_edge_diff_cpp
from compile.runner import Runner

import numpy as np


class GlowRunner(Runner):
    def __init__(self, compiler_path, data_path, mode, cal_time):
        super().__init__(compiler_path, data_path, mode, cal_time)

        cur_dir = os.path.dirname(__file__)
        if mode == 'default':
            self.run_cpp_path = os.path.join(cur_dir, "run.cpp")
            self.form_cpp = trivial_prep_run_cpp
        elif mode == 'node reduce':
            self.run_cpp_path = os.path.join(cur_dir, "node_reduce_run.cpp")
            self.form_cpp = trivial_prep_run_cpp
        else:
            self.run_cpp_path = os.path.join(cur_dir, "edge_view.cpp")
            self.form_cpp = form_edge_diff_cpp

        self.run = self.run_with_time if cal_time else self.run_without_time

    def set_input(self, data_path):
        if data_path.endswith(".bin"):
            self.data_path = data_path
        else:
            self.data_path = np_to_bin(data_path)

    def run_with_time(self, run_dir):
        run_with_time(run_dir, self.data_path)
        return self.get_run_time(run_dir)

    def run_without_time(self, run_dir):
        run_without_time(run_dir, self.data_path)

    @staticmethod
    def get_run_time(run_dir):
        return get_run_time(os.path.join(run_dir, "time.bin"))

    def compile(self, model_path, build_dir):
        r = glow_compile(self.compiler_path, model_path, build_dir)
        if r:
            raise CompilationError(model_path)
        self.form_cpp(build_dir, self.run_cpp_path)
        r = gcc_compile(build_dir)
        if r:
            raise CompilationError(model_path)

    @staticmethod
    def get_output(run_dir):
        return np.fromfile(os.path.join(run_dir, "out.bin"), dtype=np.float32)

    @staticmethod
    def get_edge_value(run_dir):
        return np.fromfile(os.path.join(run_dir, "edge.bin"), dtype=np.float32)


def glow_compile(compiler_path, model_path, build_dir):
    return os.system(
        "%s -backend=CPU -model=%s -emit-bundle=%s -network-name=model"
        % (compiler_path, model_path, build_dir))


def gcc_compile(build_dir):
    last_wd = os.getcwd()
    os.chdir(build_dir)
    os.system("g++ -c run.cpp")
    r = os.system("g++ run.o model.o -o main -no-pie")
    os.chdir(last_wd)
    return r


def trivial_prep_run_cpp(build_dir, run_cpp_path):
    shutil.copyfile(run_cpp_path, os.path.join(build_dir, "run.cpp"))


def run_with_time(run_dir, data_path):
    last_wd = os.getcwd()
    os.chdir(run_dir)
    os.system("./main %s -t" % data_path)
    os.chdir(last_wd)


def run_without_time(run_dir, data_path):
    last_wd = os.getcwd()
    os.chdir(run_dir)
    os.system("./main %s" % data_path)
    os.chdir(last_wd)


def get_run_time(time_bin_path):
    with open(time_bin_path, 'r') as f:
        return float(f.readline())


def np_to_bin(np_data_path):
    bin_data_path = os.path.join(
        os.path.dirname(os.path.abspath(np_data_path)), "data.bin")
    if not os.path.exists(bin_data_path):
        data = np.load(np_data_path)
        data.flatten().tofile(bin_data_path)
    return bin_data_path
