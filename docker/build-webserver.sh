#!/bin/sh

# cp -r ../webserver ./

sudo docker build -t mds-webserver:latest -f Dockerfile-webserver .

# rm -r ./webserver