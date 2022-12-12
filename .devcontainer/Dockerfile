########### Base Ubuntu ###########
# Base Container Microsoft: https://hub.docker.com/_/microsoft-vscode-devcontainers
# [Choice] Ubuntu version (use ubuntu-22.04 or ubuntu-18.04 on local arm64/Apple Silicon): ubuntu-22.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT=ubuntu-18.04
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

########### GLOW Compilation Preparation ###########
# Dockerfile & Tutorial: https://github.com/pytorch/glow/tree/master/utils/docker
ARG WORKDIR=/root/dev

# Create working folder
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

# Clone repo onnx2c
git clone https://github.com/kraiskil/onnx2c.git  

# Update and install tools
#RUN apt-get update && \
#    apt-get install -y clang clang-8 cmake graphviz libpng-dev \
#        libprotobuf-dev llvm-8 llvm-8-dev ninja-build protobuf-compiler wget \
#        opencl-headers libgoogle-glog-dev libboost-all-dev \
#        libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev \
#        libjemalloc-dev libpthread-stubs0-dev \
#        # Additional dependencies
#        git python-numpy && \
#    # Delete outdated llvm to avoid conflicts
#    apt-get autoremove -y llvm-6.0 && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

# Point clang to llvm-8 version
#RUN update-alternatives --install /usr/bin/clang clang \
#        /usr/lib/llvm-8/bin/clang 50 && \
#    update-alternatives --install /usr/bin/clang++ clang++ \
#        /usr/lib/llvm-8/bin/clang++ 50

# Point default C/C++ compiler to clang
#RUN update-alternatives --set cc /usr/bin/clang && \
#    update-alternatives --set c++ /usr/bin/clang++

# Install fmt
#RUN git clone https://github.com/fmtlib/fmt && \
#    mkdir fmt/build && \
#    cd fmt/build && \
#    cmake .. && make && \
#    make install

# Clean up
#RUN rm -rf fmt
