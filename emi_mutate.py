import onnx
import os

from mutation.fcb_mut import FCBMutator
from arg import init_config, mutation_args

args = init_config(mutation_args)

result_dir = os.path.join("./results", args.model_name,
                          str(args.seed_number), args.mutation_method)
print("Result saving directory is:", result_dir)

tmp_save_path = result_dir

seed_model = onnx.load(args.seed_model_path)
mutator = FCBMutator(seed_model, args.mutation_method, tmp_save_path, args.input_data_path)

mutants_saving_dir = os.path.join(result_dir, "mutants")
os.makedirs(mutants_saving_dir, exist_ok=True)

mutator.mutate(args.mutation_times, mutants_saving_dir)

os.remove(os.path.join(tmp_save_path, "tmp.onnx"))
