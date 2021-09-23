import os
import shutil
from compile.compile_err import CompilationError

import numpy as np


class GlowRunner:
    def __init__(self, compiler_path,
                 run_cpp_path=os.path.join(os.path.dirname(__file__), "glow/run.cpp"),
                 cal_time=False):
        self.compiler_path = compiler_path
        self.data_file = None
        self.run_cpp_path = run_cpp_path
        self.run = self.run_with_time if cal_time else self.run_without_time

    def set_input(self, data_file):
        if data_file.endswith(".bin"):
            self.data_file = data_file
        else:
            self.data_file = np_to_bin(data_file)

    def run_with_time(self, run_dir):
        run_with_time(run_dir, self.data_file)
        return self.get_run_time(run_dir)

    def run_without_time(self, run_dir):
        run_without_time(run_dir, self.data_file)

    @staticmethod
    def get_run_time(run_dir):
        return get_run_time(os.path.join(run_dir, "time.bin"))

    def compile(self, model_path, build_dir):
        r = glow_compile(self.compiler_path, model_path, build_dir)
        if r:
            raise CompilationError(model_path)
        gcc_compile(build_dir, self.run_cpp_path)

    @staticmethod
    def get_output(run_dir):
        return np.fromfile(os.path.join(run_dir, "out.bin"), dtype=np.float32)


def glow_compile(compiler_path, model_path, build_dir):
    return os.system(
        "%s -backend=CPU -model=%s -emit-bundle=%s -network-name=model"
        % (compiler_path, model_path, build_dir))


def gcc_compile(build_dir, run_cpp_path):
    shutil.copyfile(run_cpp_path, os.path.join(build_dir, "glow/run.cpp"))
    last_wd = os.getcwd()
    os.chdir(build_dir)
    os.system("g++ -c run.cpp")
    os.system("g++ run.o model.o -o main")
    os.chdir(last_wd)


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


def get_output(run_dir):
    return np.fromfile(os.path.join(run_dir, "out.bin"), dtype=np.float32)


def np_to_bin(np_data_path):
    bin_data_path = os.path.join(
        os.path.dirname(os.path.abspath(np_data_path)), "data.bin")
    if not os.path.exists(bin_data_path):
        data = np.load(np_data_path)
        data.flatten().tofile(bin_data_path)
    return bin_data_path
