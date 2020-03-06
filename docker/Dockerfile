#FROM tensorflow/tensorflow:2.0.1-jupyter
#FROM tensorflow/tensorflow:2.0.1-gpu-jupyter
FROM nvidia/cuda:10.1-base-ubuntu18.04
#FROM python:3.6.9

MAINTAINER maarten.schermer@naturalis.nl

COPY . /app
WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python"]
#CMD ["bash"]
