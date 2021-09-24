import os
import shutil
import onnx
from onnx import shape_inference

from arg import init_config, delta_debugging_args
from reduce.dd import JudgeFail, DeltaDebugging
from reduce.graph_applier import GraphApplier
from reduce.node_applier import make_node_applier
from compile.make_runner import make_runner


def clear_dir_content(dir_path):
    if os.path.exists(dir_path):
        os.system("rm -r %s" % dir_path)


args = init_config(delta_debugging_args)

compiler_output_dir = os.path.join(args.mutants_dir, args.compiler_name)

result_dir = os.path.join(compiler_output_dir, "reduce", "%d" % args.err_model_id)
print("Result saving directory is", result_dir)

graph_reduce_dir = os.path.join(result_dir, "graph")


def get_fault_output(runner):
    output_diff_file = os.path.join(compiler_output_dir, "output_diff.txt")
    with open(output_diff_file, 'r') as f:
        diff_line = [line for line in f.readlines()
                    if line.split("$$$")[0] == str(args.err_model_id)]
    if diff_line:
        compile_fail = False
        fault_output_dir = os.path.join(compiler_output_dir, "build", str(args.err_model_id))
        fault_output = runner.get_output(fault_output_dir)
        max_abs_diff = float(diff_line[0].split("$$$")[1])
        return compile_fail, fault_output, max_abs_diff
    else:
        compile_fail = True
        compile_err_file = os.path.join(compiler_output_dir, "compilation_failure_models.txt")
        with open(compile_err_file, 'r') as f:
            err_line = [line for line in f.readlines()
                        if os.path.splitext(os.path.basename(line.split(" $$$ ")[0]))[0] ==
                        str(args.err_model_id)][0]
        err_code = err_line.strip().split(" $$$ ")[1]
        return compile_fail, err_code


def make_judge(runner, save_dir):
    compile_fail, fail_info = get_fault_output(runner)
    if compile_fail:
        fault_output, ori_abs_diff = None, None
        err_code = fail_info
    else:
        fault_output, ori_abs_diff = fail_info
        err_code = None

    judge = JudgeFail(runner, compile_fail, save_dir,
                      input_file=args.input_data_path, err_code=err_code,
                      fault_output=fault_output, ori_abs_diff=ori_abs_diff)
    return judge


def graph_reduce():
    print("============================")
    print("Running graph reduce")
    clear_dir_content(graph_reduce_dir)
    model_dir = os.path.join(args.mutants_dir, "mutants", "models")
    mut_info_dir = os.path.join(args.mutants_dir, "mutants", "mut_info")

    runner = make_runner(args.compiler_name, args.compiler_path, args.input_data_path,
                         'default', False)

    judge = make_judge(runner, graph_reduce_dir)

    applier = GraphApplier(model_dir, mut_info_dir, args.err_model_id)

    dd = DeltaDebugging(applier, judge)
    dd.run()


def node_reduce():
    print("============================")
    print("Running node reduce")
    node_reduce_dir = os.path.join(result_dir, "node")
    clear_dir_content(node_reduce_dir)

    edges_model_dir = os.path.join(node_reduce_dir, "add_output_models")
    edges_output_dir = os.path.join(node_reduce_dir, "edge_output")
    node_reduce_run_dir = os.path.join(node_reduce_dir, "reduced_models")

    graph_reduced_model_id = max(int(model_id)
                           for model_id in os.listdir(graph_reduce_dir))
    graph_reduced_model_path = os.path.join(graph_reduce_dir, str(graph_reduced_model_id),
                                      "%d.onnx" % graph_reduced_model_id)

    applier = make_node_applier(graph_reduced_model_path, args.input_data_path,
                                edges_model_dir, edges_output_dir)

    runner = make_runner(args.compiler_name, args.compiler_path, args.input_data_path,
                         'node reduce', False)

    judge = make_judge(runner, node_reduce_run_dir)

    dd = DeltaDebugging(applier, judge)
    dd.run()

    node_reduced_model_id = max(int(model_id)
                           for model_id in os.listdir(node_reduce_run_dir))
    node_reduced_model_dir = os.path.join(node_reduce_run_dir, str(node_reduced_model_id))

    final_reduced_model_path = os.path.join(result_dir, "reduced_model")
    os.mkdir(final_reduced_model_path)

    model = onnx.load(os.path.join(node_reduced_model_dir, "%d.onnx" % node_reduced_model_id))
    model = shape_inference.infer_shapes(model)
    onnx.save(model, os.path.join(final_reduced_model_path, "reduced_model.onnx"))

    shutil.copytree(node_reduced_model_dir, os.path.join(final_reduced_model_path, "build"))


graph_reduce()
node_reduce()
