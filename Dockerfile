FROM nvidia/cuda:8.0-cudnn5-devel-centos7

MAINTAINER beetroot Project

LABEL com.nvidia.volumes.needed="nvidia_driver"

LABEL com.gpu.architecture.needed="sm30"

LABEL com.nvidia.cuda.version="8.0"

LABEL com.aws.gpu.instance="g2"

USER root

ENV PATH=/usr/local/cuda-8.0/bin:/mnt1/caffe/python:/sbin:/bin:/usr/sbin:/usr/bin:/usr/local/bin LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/include/boost:/usr/local/lib:/usr/lib64/atlas:/opt/OpenBLAS/lib:/usr/lib64 LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/include/boost:/usr/local/lib:/usr/lib64/atlas:/opt/OpenBLAS/include:/usr/lib64 PYTHONPATH=/mnt1/caffe/python:/mnt1/py-faster-rcnn/lib NLTK_DATA=/mnt1/resources/models/nltk_data THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 KERAS_BACKEND=theano

RUN apt-get -y update && \
    apt-get install -y --no-install-recommends openjdk-8-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /mnt1 && \
    cd /mnt1 && \
    yum install -y epel-release && \
    sed -i '/exclude/ s/^#*/#/' /etc/yum.conf && \
    yum -y update  && \
    yum install -y python-pip

RUN pip install theano
RUN pip install tensorflow
RUN pip install keras
RUN pip install flask
RUN pip install pyspark