FROM nvidia/cuda:8.0-cudnn5-devel-centos7

MAINTAINER beetroot Project

LABEL com.nvidia.volumes.needed="nvidia_driver"

LABEL com.gpu.architecture.needed="sm30"

LABEL com.nvidia.cuda.version="8.0"

LABEL com.aws.gpu.instance="g2"

USER root

RUN pip install theano
RUN pip install tensorflow
RUN pip install keras
RUN pip install flask
RUN pip install pyspark

RUN apt-get -y update && \
    apt-get install -y --no-install-recommends openjdk-8-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


