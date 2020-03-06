FROM tensorflow/tensorflow:2.0.1-jupyter
#FROM tensorflow/tensorflow:2.0.1-gpu-jupyter
#FROM nvidia/cuda:10.1-base-ubuntu18.04
#FROM python:3.6.9

MAINTAINER maarten.schermer@naturalis.nl

COPY . /app
WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install python3-pandas

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
#CMD ["bash"]
