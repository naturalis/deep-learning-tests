FROM python:3.6.9
MAINTAINER maarten.schermer@naturalis.nl

COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
# CMD ["app.py"]