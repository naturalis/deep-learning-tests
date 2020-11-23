#!/bin/bash

DOCKER_IMAGE=mds-tensorboard:latest

if [[ "$(docker images -q $DOCKER_IMAGE 2> /dev/null)" == "" ]]; then
    echo "container image $DOCKER_IMAGE not found"
    echo "run: ./build-tensorboard.sh"
    exit
else
    echo "got container image $DOCKER_IMAGE"
fi

function get_env {
  IFS=$'\n'
  for ITEM in $(cat .env)
  do
    LINE=$(echo $ITEM | grep "^$1=")
    if [[ ! -z "$LINE" ]]; then
    echo $LINE | sed -s 's/^[^=]*=//'
    fi
  done
}

PROJECT_ROOT=$(get_env PROJECT_ROOT)
LOG_DIR=/data/maarten.schermer${PROJECT_ROOT}log/logs_keras/*

echo "about to delete ${LOG_DIR} [y/N]"
read REPLY

if [[ "$REPLY" == "y" ]]; then
    rm -r $LOG_DIR
fi


sudo docker-compose run  --service-ports  tensorboard
