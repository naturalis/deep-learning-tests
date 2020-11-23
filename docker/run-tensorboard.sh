#!/bin/bash

CLEAR_LOGS=$1 # --clear-logs

echo "to clear old logs, run: run-tensorboard.sh --clear-logs"
echo

DOCKER_IMAGE=mds-tensorboard:latest

if [[ "$(docker images -q $DOCKER_IMAGE 2> /dev/null)" == "" ]]; then
    echo "container image $DOCKER_IMAGE not found"
    echo "run: ./build-tensorboard.sh"
    exit
else
    echo "got container image $DOCKER_IMAGE"
fi

echo

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

if [[ "$CLEAR_LOGS" == "--clear-logs" ]]; then

    PROJECT_ROOT=$(get_env PROJECT_ROOT)
    LOG_DIR=/data/maarten.schermer${PROJECT_ROOT}log/logs_keras/*

    echo "delete ${LOG_DIR} ? [y/N]"
    read REPLY

    if [[ "$REPLY" == "y" ]]; then
        rm -r $LOG_DIR
    fi
    echo
fi


echo "set up a port forward from your machine:"
echo "  ssh -L 6006:localhost:6006 <user>@<address of this server>"
echo "and open:"
echo "  http://localhost:6006/"
echo
echo "starting tensorboard"

sudo docker-compose run  --service-ports  tensorboard
