import subprocess


def execute_cmd(*command_list):
    # TODO: get rid of err model path in stderr msg.
    # TODO: only store error code
    # TODO: handle build and run TVM without input
    r = subprocess.run(list(command_list), capture_output=True)
    if r.returncode:
        return r.stderr
    else:
        return None