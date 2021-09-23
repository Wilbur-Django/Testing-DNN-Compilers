import os

import onnx
import numpy as np

from reduce.delta_split import DeltaSplitter
from reduce import reduce_utils


class GlowApplier:
    def __init__(self):
        self.save_dir = "/export/d1/dwxiao/TVM/reduced_models"
        self.compiler_path = "~/glow-build/bin/model-compiler"
        self.cpp_obj_path = "/export/d1/dwxiao/Glow/rep/build_out/27/run.o"
        self.binary_data_path = "/export/d1/dwxiao/Glow/rep/data.bin"
        self.fault_output = np.fromfile(
            "/export/d1/dwxiao/Glow/rep/build_out/27/out.bin",
            dtype=np.float32
        )

    def compile(self, model_path, build_dir):
        os.system("%s -backend=CPU -model=%s -emit-bundle=%s -network-name=model"
                  % (self.compiler_path, model_path, build_dir))

    def run(self, build_dir):
        os.system("g++ %s %s -o %s" %
                  (self.cpp_obj_path,
                   os.path.join(build_dir, "model.o"),
                   os.path.join(build_dir, "main")))
        print("=================================")
        print("Running %s:" % build_dir.split("/")[-2])
        os.system("%s %s %s %s" %
                  (os.path.join(build_dir, "main"),
                   os.path.join(build_dir, "model.weights.bin"),
                   self.binary_data_path,
                   os.path.join(build_dir, "out.bin")))

    def remain_failed(self, model, model_id):
        save_path = os.path.join(self.save_dir, "%d" % model_id)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model_path = os.path.join(save_path, "%d.onnx" % model_id)
        onnx.save(model, model_path)

        build_dir = os.path.join(save_path, "bundle")
        if not os.path.exists(build_dir):
            os.mkdir(build_dir)

        self.compile(model_path, build_dir)
        self.run(build_dir)
        return self.is_close(build_dir)

    def is_close(self, build_dir):
        model_output = np.fromfile(os.path.join(build_dir, "out.bin"),
                                   dtype=np.float32)
        diff = np.abs(model_output - self.fault_output)
        rel_diff = (diff / (self.fault_output + 1e-9))
        max_rel_diff = np.max(rel_diff)
        return max_rel_diff < 1e-6


class DeltaDebugging:
    def __init__(self, test_end, applier):
        self.ds = DeltaSplitter(test_end)
        self.ds.construct_deltas()
        self.ds.construct_dep_relation()
        dep_rel = self.ds.get_dep_relation()
        print(dep_rel)
        self.applier = applier
        self.id = 0

    def remain_failed(self, delta_ids):
        self.id += 1
        print("Applying deltas:", delta_ids)
        model = self.ds.apply_deltas(delta_ids)
        r = self.applier.remain_failed(model, self.id)
        if r:
            print("Model remain failed")
        else:
            print("Model passed")

    def resolved(self, delta_ids):
        print("Checking dependency of", delta_ids)
        r = self.ds.check_dep(delta_ids)
        if r:
            print("Check passed")
        else:
            print("Check failed")
        return r

    def apply(self, delta_ids, remaining):
        print("=============================")
        print("Delta_ids:", delta_ids)
        print("Remaining:", remaining)

        if len(delta_ids) <= 1:
            return delta_ids

        half_len = len(delta_ids) // 2
        left = delta_ids[:half_len]
        right = delta_ids[half_len:]

        left_r = reduce_utils.union_sort(left, remaining)
        right_r = reduce_utils.union_sort(right, remaining)

        if self.resolved(left_r):
            if self.remain_failed(left_r):  # Left failed
                return self.apply(left_r, remaining)  # Search left
            else:  # Left passed
                if self.resolved(right_r):
                    if self.remain_failed(right_r):  # Left passed, right failed
                        return self.apply(right_r, remaining)
                    else:  # Both passed
                        left_inducing = self.apply(left, right_r)
                        right_inducing = self.apply(right, left_r)
                        return reduce_utils.union_sort(left_inducing,
                                                       right_inducing)
                else:  # Left passed, right unresolved
                    return self.apply(right, left_r)
        else:  # Left unresolved
            # An impossible case, otherwise it requires re-partition
            raise Exception("Left cannot fail")
