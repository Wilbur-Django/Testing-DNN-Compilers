import onnx
import os

from mutation.fcb_mut import FCBMutator
from arg import init_config, mutation_args

args = init_config(mutation_args)

result_dir = os.path.join(args.result_saving_dir, args.model_name,
                          str(args.seed_number), args.mutation_method)
print("Result saving directory is:", result_dir)

tmp_save_path = result_dir

seed_model = onnx.load(args.seed_model_path)
mutator = FCBMutator(seed_model, args.mutation_method, tmp_save_path, args.input_data_path)

os.makedirs(result_dir, exist_ok=True)

mutator.mutate(args.mutation_times, result_dir)

os.remove(os.path.join(tmp_save_path, "tmp.onnx"))
