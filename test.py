import os
import shutil

ori_root_dir = "/export/d2/dwxiao/results"
new_root_dir = "/export/d2/dwxiao/build_run"

for model_name in os.listdir(ori_root_dir):
    model_dir = os.path.join(ori_root_dir, model_name)
    for seed_number in os.listdir(model_dir):
        if seed_number == "6371":
            shutil.rmtree(os.path.join(model_dir, seed_number))
            continue
        ori_build_dir = os.path.join(model_dir, seed_number, "hybrid", "tvm")
        if not os.path.exists(ori_build_dir):
            continue
        if os.path.exists(os.path.join(ori_build_dir, "build")):
            shutil.rmtree(os.path.join(ori_build_dir, "build"))
        # mutants_dir = os.path.join(model_dir, seed_number, "hybrid", "mutants", "mut_info")
        new_dir = os.path.join(new_root_dir, "tvm", model_name, seed_number, "hybrid")
        shutil.move(ori_build_dir, new_dir)
