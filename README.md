# Testing DNN Compilers
This is the research artifact for reproducing 
the results in the paper 
[Metamorphic Testing of Deep Learning Compilers](https://dl.acm.org/doi/abs/10.1145/3508035).

This repository contains the necessary materials to test deep learning compilers, including [TVM](https://tvm.apache.org/), [Glow](https://ai.meta.com/tools/glow/), and [XLA](https://www.tensorflow.org/xla).

## System requirements

Operation system: Ubuntu 18.04LTS 

CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz 16 cores

Python packages: `pip install -r requirements.txt`

## Prerequisites

**Install TVM:**

Download [TVM source code](https://github.com/apache/tvm.git) under commit 80de1239e (2021-09-25):

Follow the instructions in the [official guide](https://tvm.apache.org/docs/install/index.html) to install TVM.

**Install Glow:**

Requirements: ubuntu 18.04, LLVM 7.0.1

Download [Glow source code](https://github.com/pytorch/glow.git) under commit 0abd13adc (2021-09-21)

Follow the instructions in https://github.com/pytorch/glow to install Glow and build in release mode.

**Install XLA:**

Follow the instructions in [compile/xla/install.md](./compile/xla/install.md)

XLA version in Git commit: 7b596c44 (2021-10-03)

**Prepare seed models:**

1. Prepare a directory of your choice to hold the code, models, and data. Assume the directory is `dlcomp/`.

2. Download this repo under `dlcomp/code/`.

3. Download [the seed model](https://zenodo.org/doi/10.5281/zenodo.10852329) and extract it to `dlcomp/data/`.

4. Create directories `mutants/`, `compile_record/`, and `debugging/` under `dlcomp/` for storing the results of mutation, compilation & execution, and debugging, respectively.

## Run mutation

Go to the `dlcomp/code/` directory and run the following command to mutate the seed model:

```bash
python emi_mutate.py --model_name [model-name] --seed_model_path [path for seed ONNX model] --input_data_path [default is dlcomp/data/data.npy] --seed_number [seed_number]
```

You can find the mutated models at `dlcomp/mutants/[model-name]/[seed_number]/hybrid/models/`. The number of model names means the mutant model derives from which iteration of mutation.

## Run compilation and testing

```bash
python compile_run.py --model_name [model-name] --seed_number [seed_number] --compiler_name [compiler-name] --compiler_path [compiler_path] input_data_path [default is dlcomp/data/data.npy]
```

You can see the difference of mutants with the seed model at `dlcomp/compile_record/[compiler_name]/[model-name]/[seed_number]/hybrid/output_diff.txt`.

The second column separated by "$$$" in output_diff.txt is the max absolute difference between the prediction score of the mutant model and the seed models' prediction scores. The first column is the mutant model's id.

We regard the mutated models whose absolute difference with their seed model is greater than $10^{-4}$ as error-triggering models.
Then we reduce those models by delta-debugging
(see below).

## Run delta debugging

Assume that you have found some mutated models whose output significantly deviates from their seed model in the step [Run compilation and testing](#run-compilation-and-testing). The mutated models may be highly complex, with thousands of edges and nodes. Hence, directly debugging the mutated models to find root causes for output deviations is highly challenging. We provide a handy script to automatically shrink the size of the error-triggering mutants while still retaining the buggy behavior. The script is based on delta debugging, a standard approach to facilitating buggy input shrinking. Run the following command to execute it:

```bash
python debugging.py --model_name [model_name] --seed_number [seed_number] --compiler_name [compiler-name] --err_model_id [id number of the model you want to reduce]
```

You can see the reduced model at `dlcomp/debug/[compiler-name]/[model_name]/[seed_number]/hybrid/[err_model_id]/model.onnx`

The dumped IR when the compiler is glow is `debug_info.txt` in the same folder of the reduced model.

## How do we count bugs
Note that we regard a mutated model as
an error-triggering model if and only
if the following two conditions are
satisfied:
1. The maximum absolute difference
of its prediction score with its seed model
is greater than $10^{-4}$.
2. The model can be reduced by delta-debugging
in a reasonable time (we set 48 hours).

An error-triggering model may be due to numeric
accuracy deviation. We regard them as false
positives in our paper.

## 📜 Paper

<details><summary><b>Metamorphic Testing of Deep Learning Compilers.</b> <i>[click :: citation]</i></summary>
<div>

```bibtex
@inproceedings{10.1145/3489048.3522655,
author = {Xiao, Dongwei and Liu, Zhibo and Yuan, Yuanyuan and Pang, Qi and Wang, Shuai},
title = {Metamorphic Testing of Deep Learning Compilers},
year = {2022},
isbn = {9781450391412},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3489048.3522655},
doi = {10.1145/3489048.3522655},
booktitle = {Abstract Proceedings of the 2022 ACM SIGMETRICS/IFIP PERFORMANCE Joint International Conference on Measurement and Modeling of Computer Systems},
pages = {65–66},
numpages = {2},
keywords = {deep learning, metamorphic testing},
location = {Mumbai, India},
series = {SIGMETRICS/PERFORMANCE '22}
}
```

</div>
</details>

<p align="center">
    <a href="https://dl.acm.org/doi/10.1145/3489048.3522655"><img src="https://img.shields.io/badge/Short-2--Page_Version-4975BB.svg"></a>
    <a href="https://dl.acm.org/doi/abs/10.1145/3508035"><img src="https://img.shields.io/badge/Full-Full_Version-115878.svg"></a>
</p>
