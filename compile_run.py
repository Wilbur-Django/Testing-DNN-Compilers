import os

from arg import init_config, compilation_args

from compile.dispatch import compiler_run

args = init_config(compilation_args)

result_dir = os.path.join(args.mutants_dir, args.compiler_name)
print("Result saving directory is", result_dir)

build_dir = os.path.join(result_dir, "build")
time_record_dir = os.path.join(result_dir, "time_record")
output_diff_file = os.path.join(result_dir, "output_diff.txt")
err_file = os.path.join(result_dir, "compilation_failure_models.txt")

compiler_run(args.compiler_name, os.path.expanduser(args.compiler_path),
             os.path.join(args.mutants_dir, "mutants", "models"),
             args.input_data_path, build_dir, err_file, args.frac_compile,
             time_record_dir, output_diff_file)
