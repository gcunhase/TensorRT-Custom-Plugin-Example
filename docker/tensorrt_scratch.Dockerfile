FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG TENSORRT_VERSION=8.6.1.6
ARG CUDA_USER_VERSION=11.8
ARG CUDNN_USER_VERSION=8.9
ARG OPERATING_SYSTEM=Linux
ARG CMAKE_VERSION=3.28.0
ARG NUM_JOBS=8

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        curl \
        libjpeg-dev \
        libpng-dev \
        language-pack-en \
        locales \
        locales-all \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        libprotobuf-dev \
        protobuf-compiler \
        zlib1g-dev \
        swig \
        vim \
        gdb \
        valgrind \
        libsm6 \
        libxext6 \
        libxrender-dev && \
    apt-get clean

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip setuptools wheel

COPY ./downloads/TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz /opt
RUN cd /opt && \
    tar -xzf TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz && \
    rm TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz && \
    export PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2 | tr -d .) && \
    python3 -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt-*-cp${PYTHON_VERSION}-none-linux_x86_64.whl

ENV TRT_PATH=/opt/TensorRT-${TENSORRT_VERSION}
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TRT_PATH}/lib:${TRT_PATH}/include
ENV PATH=$PATH:${TRT_PATH}/bin

# System locale
# Important for UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
RUN rm -rf /tmp/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install cuda-python==11.8.0 \
                numpy<2 \
                onnx==1.15.0 \
                onnx-graphsurgeon \
                psutil
RUN if [ "$TENSORRT_VERSION" = "8.6.1.6" ]; then \
      pip install onnxruntime-gpu==1.17.0; \
    else \
      pip install onnxruntime-gpu==1.18.0; \
    fi \
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
