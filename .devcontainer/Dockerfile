########### Base Ubuntu ###########
# Base Container Microsoft: https://hub.docker.com/_/microsoft-vscode-devcontainers
# [Choice] Ubuntu version (use ubuntu-22.04 or ubuntu-18.04 on local arm64/Apple Silicon): ubuntu-22.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT=ubuntu-22.04
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

########### ONNX2C Preparation ###########
# Dockerfile & Tutorial: https://github.com/pytorch/glow/tree/master/utils/docker
# ARG WORKDIR=/root/dev

# Create working folder
#RUN mkdir -p $WORKDIR
#WORKDIR $WORKDIR

# Update and install tools
RUN apt-get update && apt-get install -y libprotobuf-dev protobuf-compiler cmake

# Clone repo onnx2c
RUN git clone https://github.com/kraiskil/onnx2c.git
#RUN cd /workspaces/onnx2c-minimal-example/onnx2c
#RUN git submodule update --init

# Build onnx2c
#RUN mkdir /workspaces/onnx2c-minimal-example/onnx2c/build
#RUN cd /workspaces/onnx2c-minimal-example/onnx2c/build
#RUN cmake /workspaces/onnx2c-minimal-example/onnx2c
#RUN make onnx2c
