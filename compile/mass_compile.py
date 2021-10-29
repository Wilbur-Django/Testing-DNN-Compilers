import os
import shutil

import tqdm

from compile.output_diff import array_diff, write_output_diff
from compile.make_runner import make_runner
from compile.time_utils import time_iterator
from compile.compile_err import CompilationError
from utils.path_utils import clear_and_make_dir


class MetaCompile:
    def __init__(self, compiler_name, compiler_path,
                 onnx_model_dir, result_dir, retain_result, compile_list):
        compiler_path = os.path.expanduser(compiler_path)
        self.runner = make_runner(compiler_name, compiler_path, 'default', True)
        self.onnx_model_dir = onnx_model_dir

        self.build_root_dir = os.path.join(result_dir, "build")
        time_rec_dir = os.path.join(result_dir, "time_record")
        self.compile_time_file = os.path.join(time_rec_dir, "compile_time.txt")
        self.run_time_file = os.path.join(time_rec_dir, "run_time.txt")
        self.diff_file_path = os.path.join(result_dir, "output_diff.txt")
        self.err_summary_file = os.path.join(result_dir, "compilation_failure_models.txt")
        self.err_full_info_dir = os.path.join(result_dir, "error_info")

        # self.frac_compile = frac_compile
        self.compile_list = compile_list

        self.retain_result = retain_result

        self.output = {}

        clear_and_make_dir(result_dir)
        os.mkdir(self.build_root_dir)
        os.mkdir(time_rec_dir)

    def get_compile_list(self):
        if not self.compile_list:
            model_names = [os.path.splitext(file_name)[0]
                           for file_name in os.listdir(self.onnx_model_dir)
                           if file_name != 'seed.onnx']
            model_names.sort(key=lambda x: int(x))
            # model_names = model_names[::self.frac_compile]
            model_names.append('seed')
        else:
            model_names = [str(name) for name in self.compile_list]
            if 'seed' not in model_names:
                model_names.append('seed')
        return model_names

    def handle_compilation_error(self, e: CompilationError, model_name):
        with open(self.err_summary_file, 'a') as f:
            f.write(f"{e.model_path} $$$ {e.err_code}\n")
        if not os.path.exists(self.err_full_info_dir):
            os.makedirs(self.err_full_info_dir)
        with open(os.path.join(self.err_full_info_dir, "%s.txt" % model_name), 'w') as f:
            f.write(e.err_info)


    def compile_run(self, input_file):
        model_names = self.get_compile_list()

        it = time_iterator(model_names, [self.compile_time_file, self.run_time_file])

        for model_name in tqdm.tqdm(it):
            build_dir = os.path.join(self.build_root_dir, model_name) if self.retain_result \
                else self.build_root_dir
            clear_and_make_dir(build_dir)

            failed = self.compile(model_name, build_dir, iterator=it)
            if failed:
                continue

            failed = self.run(build_dir, input_file, iterator=it)
            if failed:
                continue

            print(self.get_output(build_dir))
            self.output.update({model_name: self.get_output(build_dir)})

            if not self.retain_result:
                shutil.rmtree(build_dir)

    def compile(self, model_name, build_dir, iterator=None):
        model_path = os.path.join(self.onnx_model_dir, "%s.onnx" % model_name)
        try:
            if iterator is None:
                self.runner.build(model_path, build_dir)
            else:
                iterator.cal_time(0, lambda: self.runner.build(model_path, build_dir))
        except CompilationError as e:
            self.handle_compilation_error(e, model_name)
            shutil.rmtree(build_dir)
            return True
        else:
            return False

    def run(self, build_dir, input_file, iterator=None):
        try:
            run_time = self.runner.run_with_input(build_dir, input_file)
        except RuntimeError as e:
            with open(self.err_summary_file, 'a') as f:
                f.write(f"{build_dir} $$$ {str(e)}\n")
            shutil.rmtree(build_dir)
            return True
        else:
            if iterator is not None:
                iterator.set_time(1, run_time)
            return False

    def compare_output(self):
        name_list = [model_name for model_name in self.output.keys() if model_name != 'seed']
        name_list.sort(key=lambda x: int(x))

        seed_output = self.output['seed']

        diff_list = [array_diff(self.output[name], seed_output) for name in name_list]

        write_output_diff(self.diff_file_path, diff_list, name_list)

    def get_output(self, build_dir):
        return self.runner.get_output(build_dir)


def compiler_run(compiler_name, compiler_path, onnx_model_dir,
                 data_path, result_dir, retain_result, compile_list):
    meta_compiler = MetaCompile(compiler_name, compiler_path,
                                onnx_model_dir, result_dir, retain_result, compile_list)
    meta_compiler.compile_run(data_path)
    meta_compiler.compare_output()
