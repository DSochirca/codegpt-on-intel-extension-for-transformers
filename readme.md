# CodeGPT on Intel
_This repository is part of the 2023 [Research Project](https://github.com/TU-Delft-CSE/Research-Project) of [TUDelft](https://https//github.com/TU-Delft-CSE)._

## Description

Here we present the scripts required to reproduce our experiments performed for the bachelor thesis: [_Compressing Code Generation Language Models on CPUs: Using Group Lasso Pruning and Post-training Quantization_](https://repository.tudelft.nl/islandora/object/uuid:47817baa-9c64-4cca-b206-09544ac5a75b?collection=education).

In this codebase we compress the [CodeGPT](https://huggingface.co/AISE-TUDelft/CodeGPT-Py150/tree/main) model, which we fine-tuned on the [Py150 dataset](https://huggingface.co/datasets/0n1xus/codexglue). We use the [intel-extension-for-transformers toolkit](https://github.com/intel/intel-extension-for-transformers/tree/main) to apply Group Lasso pruning and Post-training Quantization to the model.

## Features

- Scripts to apply Group Lasso pruning to CodeGPT at different sparsity levels. More info about pruning at [intel's pruning doc](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/pruning.md).
- Scripts for applying dynamic post-training quantization. More info about quantization at [intel's quantization doc](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/quantization.md).
- Evaluation scripts to measure performance in terms of model size, CPU inference speed, and accuracy metrics.

## Contents

- `/outputs`: Output files from running the scripts
- `/results`: Evaluation results from our experiments
- `compression_*.py`: Scripts for pruning and quantization
- `environment.yml`: Conda environment file
- `evaluation.py`: Script for evaluating the compressed models
- `example_bash_script.sh`: Example bash script for running the python code
- `vizualization.ipynb`: Jupyter notebook to generate plots and run our onnx experiment

## Installation and Usage

See our [example_bash_script.sh](https://github.com/AISE-TUDelft/LLM4CodeCompression/blob/intel/CodeGPT-on-Intel/example_bash_script.sh) for an example of how to set up the environment and run python files.

## The Resultant Compressed Models

Original model: [AISE-TUDelft/CodeGPT-Py150](https://huggingface.co/AISE-TUDelft/CodeGPT-Py150) <br>

Pruned models: <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.4-sparsity](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.4-sparsity) <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.5-sparsity](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.5-sparsity) <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.6-sparsity](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.6-sparsity) <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.7-sparsity](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.7-sparsity) <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.8-sparsity](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.8-sparsity) <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.9-sparsity](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.9-sparsity) <br>

Quantized models (weights + activations): <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-asym-per-channel](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-asym-per-channel) <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-sym-per-channel](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-sym-per-channel) <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-asym-per-tensor](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-asym-per-tensor) <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-sym-per-tensor](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-sym-per-tensor) <br>

Quantized models (only weights): <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-asym-per-channel](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-asym-per-channel) <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-sym-per-channel](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-sym-per-channel) <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-asym-per-tensor](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-asym-per-tensor) <br>
[AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-sym-per-tensor](https://huggingface.co/AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-sym-per-tensor) <br>
