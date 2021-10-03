import os


class Runner:
    def __init__(self, compiler_path, data_path, mode, cal_time):
        self.compiler_path = compiler_path
        self.data_path = os.path.abspath(data_path)
        self.cal_time = cal_time
        self.mode = mode

    def set_input(self, data_path):
        self.data_path = os.path.abspath(data_path)

    def compile_run(self, model_path, build_dir, view_edge=False):
        self.compile(model_path, build_dir)
        self.run(build_dir)
        if view_edge:
            return self.get_edge_value(build_dir), self.get_output(build_dir)
        else:
            return self.get_output(build_dir)


    def compile(self, model_path, build_dir):
        raise NotImplementedError()

    def run(self, run_dir):
        raise NotImplementedError()

    @staticmethod
    def get_output(run_dir):
        raise NotImplementedError()

    @staticmethod
    def get_edge_value(run_dir):
        raise NotImplementedError()
