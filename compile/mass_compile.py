import os
import shutil

import tqdm

from compile.output_diff import array_diff, write_output_diff
from compile.make_runner import make_runner
from compile.time_utils import time_iterator
from compile.compile_err import CompilationError


class MetaCompile:
    def __init__(self, runner, onnx_model_dir, build_root_dir,
                 diff_file_path, time_record_dir,
                 err_summary_file, err_full_info_dir,
                 frac_compile, compile_list):
        self.runner = runner
        self.onnx_model_dir = onnx_model_dir
        self.build_root_dir = build_root_dir
        self.time_rec_dir = time_record_dir
        self.diff_file_path = diff_file_path
        self.err_summary_file = err_summary_file
        self.err_full_info_dir = err_full_info_dir
        self.frac_compile = frac_compile
        self.compile_list = compile_list

        if os.path.exists(build_root_dir):
            shutil.rmtree(build_root_dir)
        os.makedirs(build_root_dir)

        if os.path.exists(err_summary_file):
            os.remove(err_summary_file)

        if os.path.exists(err_full_info_dir):
            shutil.rmtree(err_full_info_dir)
        os.makedirs(err_full_info_dir)

    def get_compile_list(self):
        if not self.compile_list:
            model_names = [os.path.splitext(file_name)[0]
                           for file_name in os.listdir(self.onnx_model_dir)
                           if file_name != 'seed.onnx'][::self.frac_compile]
            model_names.append('seed')
        else:
            model_names = [str(name) for name in self.compile_list]
            if 'seed' not in model_names:
                model_names.append('seed')
        return model_names

    def compile(self):
        time_file = os.path.join(self.time_rec_dir, "compile_time.txt")

        model_names = self.get_compile_list()

        it = time_iterator(model_names, time_file)

        for model_name in tqdm.tqdm(it):
            model_path = os.path.join(self.onnx_model_dir, "%s.onnx" % model_name)
            build_dir = os.path.join(self.build_root_dir, model_name)
            os.makedirs(build_dir, exist_ok=True)
            try:
                it.cal_time(lambda: self.runner.compile(model_path, build_dir))
            except CompilationError as e:
                shutil.rmtree(build_dir)
                with open(self.err_summary_file, 'a') as f:
                    f.write(f"{e.model_path} $$$ {e.err_code}\n")
                with open(os.path.join(self.err_full_info_dir, "%s.txt" % model_name), 'w') as f:
                    f.write(e.err_info)

    def run(self, input_file):
        time_file = os.path.join(self.time_rec_dir, "run_time.txt")

        it = time_iterator(os.listdir(self.build_root_dir), time_file)

        self.runner.set_input(input_file)

        for dir_name in it:
            run_dir = os.path.join(self.build_root_dir, dir_name)
            try:
                run_time = self.runner.run(run_dir)
            except RuntimeError as e:
                shutil.rmtree(run_dir)
                with open(self.err_summary_file, 'a') as f:
                    f.write(f"{run_dir} $$$ {str(e)}\n")
                continue
            it.set_time(run_time)

    def compare_output(self):
        name_list = [dir_name for dir_name in os.listdir(self.build_root_dir)
                     if dir_name != 'seed']
        name_list.sort(key=lambda x: int(x))

        seed_output = self.get_output('seed')

        diff_list = [array_diff(self.get_output(name), seed_output) for name in name_list]

        write_output_diff(self.diff_file_path, diff_list, name_list)

    def get_output(self, model_name):
        return self.runner.get_output(os.path.join(self.build_root_dir, model_name))


def compiler_run(compiler_name, compiler_path, onnx_model_dir,
                 data_path, build_dir,
                 err_summary_file, err_full_info_dir,
                 time_record_dir, diff_file_path,
                 frac_compile, compile_list):
    os.makedirs(time_record_dir, exist_ok=True)
    os.makedirs(build_dir, exist_ok=True)
    runner = make_runner(compiler_name, compiler_path, data_path, 'default', True)

    meta_compiler = MetaCompile(runner, onnx_model_dir, build_dir,
                                diff_file_path, time_record_dir,
                                err_summary_file, err_full_info_dir,
                                frac_compile, compile_list)
    meta_compiler.compile()
    meta_compiler.run(data_path)
    meta_compiler.compare_output()
