# Testing DNN Compilers
This is the artifact for reproducing 
the results in the paper 
"Metamorphic Testing of Deep Learning Compilers" published in [SIGMETRICS 2022](https://sigmetrics.org/sigmetrics2022/). If you find it useful, please consider [citing our paper](#-paper).

Interested readers may follow the steps below to 
run mutation, compilation, and 
delta-debugging to reproduce the results in
that paper.

## System requirement

Operation system: Ubuntu 18.04LTS 

CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz 16 cores

Python package dependency (installed in conda environment):

onnx: 1.10.1

onnxruntime: 1.9.0

tqdm

## Installing TVM

Follow the instructions in https://tvm.apache.org/docs/install/index.html to install the TVM in conda environment

TVM version in Git commit: 80de1239e (2021-09-25)

## Installing Glow

Follow the instructions in https://github.com/pytorch/glow to install Glow. Built in release mode.

Glow version in Git commit: 0abd13adc (2021-09-21)

Requirement: ubuntu 18.04, LLVM 7.0.1

## Installing XLA

Follow the instructions in compile/xla/install.md

XLA version in Git commit: 7b596c44 (2021-10-03)



## Preparing the folder structure

Download this code. At the parent directory-level of the code directory, download https://drive.google.com/file/d/1ODEKR016GTJLeHemgZ_JuIBoBz4AWtHC/view?usp=sharing and unzip the data.zip to data/. 

Also, at the parent directory-level of the code, make directories `mutants/`, `compile_record/`, `debugging/`, for storing the results for mutation, compilation & running, and debugging, respectively.



## Run mutation

```shell
python emi_mutate.py --model_name [model-name] --seed_model_path [path for seed ONNX model] --input_data_path [default is ../data/data.npy] --seed_number [seed_number]
```

You can find the mutated models at `../mutants/[model-name]/[seed_number]/hybrid/models/`. The number of model names means the mutant model derives from which iteration of mutation.

## Run compilation & running

```shell
python compile_run.py --model_name [model-name] --seed_number [seed_number] --compiler_name [compiler-name] --compiler_path [compiler_path] input_data_path [default is ../data/data.npy]
```

You can see the difference of mutants with seed model at `../compile_record/[compiler_name]/[model-name]/[seed_number]/hybrid/output_diff.txt`.

The second column separated by "$$$" in output_diff.txt is the max absolute difference for the prediction score of the mutant model and the seed model. The first column is the mutant model's id.

We regard the mutated models whose absolute
difference with their seed model greater than
10^-4 as error-triggering models.
Then we reduce those models by delta-debugging
(see below).

## Run delta debugging

```shell
python debugging.py --model_name [model_name] --seed_number [seed_number] --compiler_name [compiler-name] --err_model_id [id number of the model you want to reduce]
```

You can see the reduced model at `../debug/[compiler-name]/[model_name]/[seed_number]/hybrid/[err_model_id]/model.onnx`

The dumped IR when the compiler is glow is `debug_info.txt` at same folder of the reduced model.

## How we count bugs
Note that we regard a mutated model as
an error-triggering models if and only
if the following two conditions are
satisfied:
1. The maximum absolute difference
of its prediction score with its seed model
is greater than 10^-4.
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
    <a href="https://dl.acm.org/doi/10.1145/3489048.3522655"><img src="https://img.shields.io/badge/Paper-SIGMETRICS'22-4975BB.svg"></a>
    <a href="https://dl.acm.org/doi/abs/10.1145/3508035"><img src="https://img.shields.io/badge/Full-Full_Version-115878.svg"></a>
</p>
