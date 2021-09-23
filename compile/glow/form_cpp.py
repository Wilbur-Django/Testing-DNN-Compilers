import re
import os


def get_header_info(header_file):
    with open(header_file, "r") as f:
        lines = [line for line in f.readlines()]
        for i, line in enumerate(lines):
            m = re.match(r"\/\/   Name: \"(A(\d+)(__\d)?)\"", line.strip())
            if m:
                edge_macro = m.group(1)
                break
        for line in lines[i + 1:]:
            m = re.match(r"\/\/   Size: (\d+) \(bytes\)", line.strip())
            if m:
                edge_size = m.group(1)
                break
    return edge_macro, edge_size


def make_run_cpp(ori_run_file, new_run_file, header_file):
    with open(ori_run_file, 'r') as f:
        lines = f.readlines()
    edge_macro, edge_size = get_header_info(header_file)
    lines = [line.replace("MODEL_A126", "MODEL_%s" % edge_macro)
                 .replace('262144', edge_size) for line in lines]

    with open(new_run_file, 'w') as f:
        f.write("".join(lines))


def form_edge_diff_cpp(build_dir, ori_run_file):
    header_file = os.path.join(build_dir, "model.h")
    new_run_file = os.path.join(build_dir, "run.cpp")
    make_run_cpp(ori_run_file, new_run_file, header_file)
