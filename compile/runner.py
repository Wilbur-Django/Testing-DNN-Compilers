class Runner:
    def __init__(self, compiler_path, data_path, mode, cal_time):
        self.compiler_path = compiler_path
        self.data_path = data_path
        self.cal_time = cal_time
        self.mode = mode

    def set_input(self, data_path):
        self.data_path = data_path

    def compile(self, model_path, build_dir):
        raise NotImplementedError()

    @staticmethod
    def get_output(run_dir):
        raise NotImplementedError()