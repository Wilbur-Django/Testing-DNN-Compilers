import subprocess


def execute_cmd(*command_list):
    r = subprocess.run(list(command_list), capture_output=True)
    if r.returncode:
        if r.returncode == -11:
            return "Segmentation fault"
        return r.stderr
    else:
        return None


def write_in_out_info(file, onnx_model):
    if len(onnx_model.graph.input) > 0:
        has_input = 1
    else:
        has_input = 0
    if len(onnx_model.graph.output) > 1:
        has_two_output = 1
    else:
        has_two_output = 0
    with open(file, 'w') as f:
        f.write(f"{has_input}{has_two_output}")


def read_in_out_info(file):
    with open(file, 'r') as f:
        line = f.readline()
    has_input, has_two_output = bool(int(line[0])), bool(int(line[1]))
    return has_input, has_two_output