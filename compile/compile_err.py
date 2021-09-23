class CompilationError(Exception):
    def __init__(self, model_path, err_info='None'):
        self.model_path = model_path
        self.err_info = err_info

    def __str__(self):
        return "Compilation failed: %s $$$$$ %s" % (self.model_path, self.err_info)
