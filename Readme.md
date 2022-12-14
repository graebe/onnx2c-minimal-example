# ONNX2C Minimal Examples

This repository implements minimal examples for the python package onnx2c (https://github.com/kraiskil/onnx2c). You can use it as an implementation example and get a feeling for the execution performance of the compiled code. I will also include my own benchmarks in the Readme.md.

## Prerequisites

This repo has been developed in a docker container specified in the folder .devcontainer. You can run it locally in visual studio code with the VSC Container Dev extension (Docker is prerequisite). Another option is to use github codespaces.

## Run

Use the scripts to generate and compile reference models (*_build.sh). They perform the following steps:

1. Instantiate a Pytorch Model
2. Export to ONNX Format
3. Translate to C-Code with onnx2c
4. Execute Benchmarks and print results

## Benchmarks

The benchmarks were performed on a Macbook Pro 2021 (M1).

| Model                  |Wall Time Pytorch      |  Wall Time ONNX  |  Wall Time C    |
|:-----------------------|----------------------:|-----------------:|----------------:|
|Feed-Forward NN.        |           3908.99 mus |       498.70 mus |       36.00 mus |
|MobilnetV2              |          34766.20 mus |     23331.70 mus |  2104377.00 mus |

Compilations for real-time targets (with other models) have been performed in the original repo (https://github.com/kraiskil/onnx2c).
