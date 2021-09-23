def make_runner(compiler_name, compiler_path, data_path, mode, cal_time):
    if compiler_name == 'glow':
        from compile.glow.glow import GlowRunner
        return GlowRunner(compiler_path, data_path, mode, cal_time)
    elif compiler_name == 'tvm':
        from compile.tvm import TVMRunner
        return TVMRunner(compiler_path, data_path, mode, cal_time)
    elif compiler_name == 'xla':
        from compile.xla.xla import XlaRunner
        return XlaRunner(compiler_path, data_path, mode, cal_time)
    else:
        return None