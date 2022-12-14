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
|Feed-Forward NN         |           2515.45 mus |       388.04 mus |       18.00 mus |
|CNN_minimal             |           1054.54 mus |       268.37 mus |      215.00 mus |
|CNN_3layers             |          15042.33 mus |     11603.04 mus |   847211.00 mus |
|MobilnetV2              |          25823.16 mus |     17536.99 mus |  2033580.00 mus |

Compilations for real-time targets (with other models) have been performed in the original repo (https://github.com/kraiskil/onnx2c).
