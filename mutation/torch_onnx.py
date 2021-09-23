import os
import random

import torch

from mutation.fcb import mutate
from mutation.utils import path_append_timestamp

import numpy as np

from PIL import Image
import torchvision.transforms as transforms
import onnx
import onnxruntime

from tvm.contrib.download import download_testdata

onnx_model = onnx.load("./saved_models/super_resolution.onnx")

# onnx.checker.check_model(onnx_model)
# print(onnx.helper.printable_graph(onnx_model.graph))

save_path = path_append_timestamp("mutation/mutated_models/super_resolution/")

random.seed(1)

mutate(onnx_model, 10, save_path)


def rand_test(test_times, ort_session_ori, ort_session_mut):
    for i in range(1, test_times + 1):
        np_input = np.random.randn(1, 1, 224, 224).astype('float32')
        inputs = {ort_session_ori.get_inputs()[0].name: np_input}
        cmp_mut_out(ort_session_ori,
                    ort_session_mut,
                    inputs)
        print("Test %d passed" % i)
    print("Testing passed")


def cmp_mut_out(ort_session_ori, ort_session_mut, inputs):
    out_ori = ort_session_ori.run(None, inputs)
    out_mut = ort_session_mut.run(None, inputs)
    np.testing.assert_allclose(out_ori[0], out_mut[0])


def fcb_test(test_times, mut_path):
    ori_file = "./saved_models/super_resolution.onnx"
    ort_session_ori = onnxruntime.InferenceSession(ori_file)
    for i, mut_file in enumerate(os.listdir(mut_path)):
        ort_session_mut = onnxruntime.InferenceSession(os.path.join(mut_path, mut_file))
        print("Testing file", mut_file)
        rand_test(test_times, ort_session_ori, ort_session_mut)


fcb_test(1, save_path)

exit(2)

ort_session_before = onnxruntime.InferenceSession("../saved_models/super_resolution.onnx")

# compute ONNX Runtime output prediction
ort_inputs = None
ort_outs = ort_session_before.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
torch_out = torch.randn(1, 1, 224, 224)
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")

img = Image.open(img_path)

resize = transforms.Resize([224, 224])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)
ort_inputs = {ort_session_before.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session_before.run(None, ort_inputs)
img_out_y = ort_outs[0]
img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# Save the image, we will compare this with the output image from mobile device
final_img.save("./saved_models/cat_superres_with_ort.jpg")
