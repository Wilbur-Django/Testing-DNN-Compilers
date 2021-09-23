import os

import onnx
import numpy as np

from reduce import reduce_utils


class JudgeFail:
    def __init__(self, runner, compile_fail, save_dir,
                 err_output_dir=None, input_file=None, threshold=0.1):
        self.save_dir = save_dir
        self.runner = runner
        self.id = 0
        self.compile_fail = compile_fail

        self.runner.set_input(input_file)

        if not compile_fail:
            self.fault_output = self.runner.get_output(err_output_dir)
            self.threshold = threshold

    def compile(self, model_path, build_dir):
        self.runner.compile(model_path, build_dir)

    def run(self, run_dir):
        # self.runner.set_input(self.input_file)
        self.runner.run(run_dir)

    def remain_failed(self, model):
        self.id += 1
        print("=================================")
        print("Running %s:" % self.id)
        save_path = os.path.join(self.save_dir, "%d" % self.id)
        os.makedirs(save_path, exist_ok=True)

        model_path = os.path.join(save_path, "%d.onnx" % self.id)
        onnx.save(model, model_path)

        build_dir = os.path.join(save_path, "build")
        os.makedirs(build_dir, exist_ok=True)

        if self.compile_fail:
            try:
                self.compile(model_path, build_dir)
            except Exception:
                return True
            return False

        self.compile(model_path, build_dir)
        self.run(build_dir)
        return self.is_close(build_dir)

    def is_close(self, run_dir):
        model_output = self.runner.get_output(run_dir)
        diff = np.abs(model_output - self.fault_output)
        rel_diff = diff / (
                np.abs(self.fault_output) + np.abs(model_output) + 1e-9)
        max_rel_diff = np.max(rel_diff)
        print("Max relative diff is %f" % max_rel_diff)
        return max_rel_diff < self.threshold


class DeltaDebugging:
    def __init__(self, applier, judge):
        self.applier = applier
        self.judge = judge

    def remain_failed(self, delta_ids):
        print("Applying deltas:", delta_ids)
        model = self.applier.apply(delta_ids)
        r = self.judge.remain_failed(model)
        if r:
            print("Model remain failed")
        else:
            print("Model passed")
        return r

    def resolved(self, delta_ids):
        return True
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

        print("Left:", left)
        print("Right:", right)

        left_r = reduce_utils.union_sort(left, remaining)
        right_r = reduce_utils.union_sort(right, remaining)

        if self.resolved(left_r):
            if self.remain_failed(left_r):  # Left failed
                return self.apply(left, remaining)  # Search left
            else:  # Left passed
                if self.resolved(right_r):
                    if self.remain_failed(right_r):  # Left passed, right failed
                        return self.apply(right, remaining)
                    else:  # Both passed
                        left_inducing = self.apply(left, right_r)
                        left_r = reduce_utils.union_sort(
                            left_inducing, remaining)
                        right_inducing = self.apply(right, left_r)
                        return reduce_utils.union_sort(left_inducing,
                                                       right_inducing)
                else:  # Left passed, right unresolved
                    return self.apply(right, left_r)
        else:  # Left unresolved
            # An impossible case, otherwise it requires re-partition
            raise Exception("Left cannot fail")

    def run(self):
        error_inducing = self.apply(list(self.applier.valid_range()), [])
        self.apply_err_inducing(error_inducing)

    def apply_err_inducing(self, error_inducing):
        print("Error-inducing deltas:", error_inducing)
        dep_chain = error_inducing
        # dep_chain = self.ds.get_dep_chain(error_inducing)
        # print("Error-inducing deltas with their dependencies", dep_chain)
        self.remain_failed(dep_chain)
