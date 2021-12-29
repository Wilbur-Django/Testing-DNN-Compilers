import os
import numpy as np


class Counter:
    def __init__(self):
        self.diff_list = []
        self.a_greater_b = 0
        self.a_less_b = 0
        self.a_same_b = 0
        self.not_comparable = 0
        self.threshold = 1e-6

    def add(self, a, b, name):
        if a is None or b is None:
            self.not_comparable += 1
            self.diff_list.append((name, "Not comparable"))
            return
        diff = a - b
        if diff > self.threshold:
            self.a_greater_b += 1
        elif diff < -self.threshold:
            self.a_less_b += 1
        else:
            self.a_same_b += 1
        self.diff_list.append((name, diff))

    def __str__(self):
        return f"greater: {self.a_greater_b}, same: {self.a_same_b}, less: {self.a_less_b}"

def load_output(record_dir, seed_model, seed_n, model_id):
    output_path = os.path.join(
        record_dir, seed_model, seed_n, "hybrid", "output", f"{model_id}.npy")
    if not os.path.exists(output_path):
        return None
    return np.load(output_path)

def get_max_abs_diff(a, b):
    return np.max(np.abs(a - b))

compile_record_dir = "../compile_record"
glow_record_dir = os.path.join(compile_record_dir, "glow")
tvm_record_dir = os.path.join(compile_record_dir, "tvm")
tf_record_dir = os.path.join(compile_record_dir, "tensorflow")

compared_mutants = [
    ("resnet18", "10486859"),
    ("vgg11", "97"),
    ("mobilenet", "10486859"),
    ("mobilenet", "99131411")
]

o1_o3_cnt, o2_o3_cnt = Counter(), Counter()

for seed_model_name, seed_number in compared_mutants:
    glow_seed_out = load_output(glow_record_dir, seed_model_name, seed_number, "seed")
    for i in range(1, 1001, 10):
        glow_out = load_output(glow_record_dir, seed_model_name, seed_number, i)
        tvm_out = load_output(tvm_record_dir, seed_model_name, seed_number, i)
        tf_out = load_output(tf_record_dir, seed_model_name, seed_number, i)

        if glow_out is None:
            continue

        o3_diff = get_max_abs_diff(glow_out, glow_seed_out)
        if tvm_out is not None:
            o1_diff = get_max_abs_diff(glow_out, tvm_out)
        else:
            o1_diff = None

        if tf_out is not None:
            o2_diff = get_max_abs_diff(glow_out, tf_out)
        else:
            o2_diff = None

        o1_o3_cnt.add(o1_diff, o3_diff, f"{seed_model_name}, {seed_number}, {i}")
        o2_o3_cnt.add(o2_diff, o3_diff, f"{seed_model_name}, {seed_number}, {i}")

print("o1 o3", o1_o3_cnt)
print("o2 o3", o2_o3_cnt)
