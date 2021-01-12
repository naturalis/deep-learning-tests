FROM tensorflow/tensorflow:2.0.3-gpu-py3-jupyter
#FROM tensorflow/tensorflow:2.0.1-gpu-py3-jupyter

MAINTAINER maarten.schermer@naturalis.nl

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
#CMD ["bash"]




