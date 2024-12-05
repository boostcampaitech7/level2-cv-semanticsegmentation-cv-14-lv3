FROM nvidia/cuda:12.1.0-base-ubuntu20.04
MAINTANER Namgyu-Youn <yynk2012@gmail.com>


ARG PYTHON
ENV PYTHON=$PYTHON
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libgl1-mesa-glx \

# Install required packeages
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt