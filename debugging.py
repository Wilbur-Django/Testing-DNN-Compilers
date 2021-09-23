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


# def make_runner(is_graph_reduce):
#     if args.compiler_name == 'glow':
#         from compile.glow.glow import GlowRunner
#         if is_graph_reduce:
#             runner = GlowRunner(os.path.expanduser(args.compiler_path))
#         else:
#             runner = GlowRunner(os.path.expanduser(args.compiler_path), mode='node reduce')
#     elif args.compiler_name == 'tvm':
#         from compile.tvm import TVMRunner
#         runner = TVMRunner(args.input_data_path)
#     elif args.compiler_name == 'xla':
#         from compile.xla.xla import XlaRunner
#         runner = XlaRunner(os.path.expanduser(args.compiler_path), cal_time=False)
#     elif args.compiler_name == 'onnx':
#         from compile.onnx_runner.onnx_runner import OnnxRunner
#         runner = OnnxRunner(args.compiler_path)
#     else:
#         raise Exception("Compiler name not supported")
#
#     return runner

# def get_fault_output(runner):
compile_failed_log = os.path.join(compiler_output_dir, "")
err_output_dir = os.path.join(compiler_output_dir, "build", "%d" % args.err_model_id)


def graph_reduce():
    print("============================")
    print("Running graph reduce")
    clear_dir_content(graph_reduce_dir)
    model_dir = os.path.join(args.mutants_dir, "mutants", "models")
    mut_info_dir = os.path.join(args.mutants_dir, "mutants", "mut_info")

    runner = make_runner(args.compiler_name, args.compiler_path, args.input_data_path,
                         'default', False)

    judge = JudgeFail(runner, args.compile_fail, graph_reduce_dir,
                      err_output_dir=err_output_dir,
                      input_file=args.input_data_path)

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

    reduced_model_id = max(int(model_id)
                           for model_id in os.listdir(graph_reduce_dir))
    reduced_model_path = os.path.join(graph_reduce_dir, str(reduced_model_id),
                                      "%d.onnx" % reduced_model_id)

    applier = make_node_applier(reduced_model_path, args.input_data_path,
                                edges_model_dir, edges_output_dir)

    runner = make_runner(args.compiler_name, args.compiler_path, args.input_data_path,
                         'node reduce', False)

    judge = JudgeFail(runner, args.compile_fail, node_reduce_run_dir,
                      err_output_dir=err_output_dir,
                      input_file=args.input_data_path)

    dd = DeltaDebugging(applier, judge)
    dd.run()

    reduced_model_id = max(int(model_id)
                           for model_id in os.listdir(node_reduce_run_dir))
    model = onnx.load(os.path.join(node_reduce_run_dir,
                                 str(reduced_model_id),
                                 "%d.onnx" % reduced_model_id))
    model = shape_inference.infer_shapes(model)
    onnx.save(model, os.path.join(result_dir, "reduced_model.onnx"))
    # shutil.copyfile(os.path.join(node_reduce_run_dir,
    #                              str(reduced_model_id),
    #                              "%d.onnx" % reduced_model_id),
    #                 os.path.join(result_dir, "reduced_model.onnx"))


graph_reduce()
node_reduce()
