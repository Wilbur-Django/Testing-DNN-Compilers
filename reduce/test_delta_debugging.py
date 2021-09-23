import onnx

from reduce.delta_debugging import GlowApplier, DeltaDebugging


def remain_failed():
    ga = GlowApplier()
    model = onnx.load("/export/d1/dwxiao/TVM/mutation/mutated_models/resnet18/0526_085234/27.onnx")
    if ga.remain_failed(model, 1):
        print("Failed")
    else:
        print("Passed")


# remain_failed()

test_end = 27
ga = GlowApplier()
dd = DeltaDebugging(test_end, ga)
error_inducing = dd.apply(list(range(1, test_end + 1)), [])
print(error_inducing)
