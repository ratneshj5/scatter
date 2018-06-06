# Extension of the Jupyter Notebooks
# Distributed under the terms of the Modified BSD / MIT License.
FROM jupyter/scipy-notebook

MAINTAINER beetroot Project

USER root

RUN pip install dist-keras

RUN pip install pyspark

RUN apt-get -y update && \
    apt-get install -y --no-install-recommends openjdk-8-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


