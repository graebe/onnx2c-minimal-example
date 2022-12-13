########### Base Ubuntu ###########
# Base Container Microsoft: https://hub.docker.com/_/microsoft-vscode-devcontainers
# [Choice] Ubuntu version (use ubuntu-22.04 or ubuntu-18.04 on local arm64/Apple Silicon): ubuntu-22.04, ubuntu-20.04, ubuntu-18.04
FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

########### ONNX2C Preparation ###########
# Update and install tools
RUN apt-get update && apt-get install -y libprotobuf-dev protobuf-compiler cmake

# Clone repo onnx2c
RUN git clone https://github.com/kraiskil/onnx2c.git
RUN cd onnx2c && git submodule update --init

# Build onnx2c
RUN mkdir /onnx2c/build && cd /onnx2c/build && cmake /onnx2c && make onnx2c

# Install python packages
RUN pip install -r 