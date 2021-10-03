from utils.path_utils import norm_user_path

def make_runner(compiler_name, compiler_path, data_path, mode, cal_time):
    compiler_path = norm_user_path(compiler_path)
    if compiler_name == 'glow':
        from compile.glow.glow import GlowRunner
        return GlowRunner(compiler_path, data_path, mode, cal_time)
    elif compiler_name == 'tvm':
        from compile.tvm.tvm import TVMRunner
        return TVMRunner(compiler_path, data_path, mode, cal_time)
    elif compiler_name == 'xla':
        from compile.xla.xla import XlaRunner
        return XlaRunner(compiler_path, data_path, mode, cal_time)
    else:
        return None