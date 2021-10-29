import os

from arg import init_config, compilation_args

from compile.mass_compile import compiler_run

args = init_config(compilation_args)

mutant_models_dir = os.path.join(args.mutants_dir, args.model_name, str(args.seed_number),
                                 args.mutation_method, "models")

result_dir = os.path.join(args.compile_record_dir, args.compiler_name,
                          args.model_name, str(args.seed_number),
                          args.mutation_method)
print("Result saving directory is", result_dir)

compiler_run(args.compiler_name, args.compiler_path,
             mutant_models_dir, args.input_data_path, result_dir,
             args.retain_result, args.compile_list)
