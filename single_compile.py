import os

from compile.make_runner import make_runner

# build_dir = "/export/d2/dwxiao/temp"
# model_path = "/export/d2/dwxiao/mutants/mobilenet/1/hybrid/mutants/models/11.onnx"
# model_path = "/export/d2/dwxiao/data/mobilenet.onnx"
# model_path = "/export/d2/dwxiao/mutants/resnet18/1/hybrid/models/21.onnx"

# compiler_name = "tvm"
# compiler_path = "/export/d2/dwxiao/tvm"

# compiler_name = "xla"
# compiler_path = "/export/d2/dwxiao/tensorflow"

# compiler_name = "glow"
# compiler_path = "/export/d2/dwxiao/build_Release/bin/model-compiler"

compiler_name = 'xla'
compiler_path = "/root/tensorflow"
model_path = "/export/mutants/mobilenet/1/hybrid/models/seed.onnx"
build_dir = "/export/temp"

runner = make_runner(compiler_name, compiler_path, "../data/data.npy",
                     'default', False)

runner.compile(model_path, build_dir)
runner.run(build_dir)
print(runner.get_output(build_dir))
