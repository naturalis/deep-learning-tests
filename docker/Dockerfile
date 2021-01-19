FROM tensorflow/tensorflow:2.2.2-gpu-py3-jupyter
#FROM tensorflow/tensorflow:2.0.1-gpu-py3-jupyter

# 2.0.1 -> 2.2.2: 01.2021 - 2.0.1 crashes w/o error while filling shuffle buffer when upsampling    
#https://hub.docker.com/r/tensorflow/tensorflow/tags

MAINTAINER maarten.schermer@naturalis.nl

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
#CMD ["bash"]




