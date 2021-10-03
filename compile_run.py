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

build_dir = os.path.join(result_dir, "build")
time_record_dir = os.path.join(result_dir, "time_record")
output_diff_file = os.path.join(result_dir, "output_diff.txt")

err_summary_file = os.path.join(result_dir, "compilation_failure_models.txt")
err_full_info_dir = os.path.join(result_dir, "error_info")

compiler_run(args.compiler_name, os.path.expanduser(args.compiler_path),
             mutant_models_dir,
             args.input_data_path, build_dir,
             err_summary_file, err_full_info_dir,
             time_record_dir, output_diff_file,
             args.frac_compile, args.compile_list)
