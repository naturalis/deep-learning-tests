#!/bin/bash

if [[ -z "$1" ]]; then
    echo "no script to run"
    exit
fi

SCRIPT_TO_RUN=$1

if [[ ! -z "$2" ]]; then
    ENV="--env-file $2"
elif [[ -f ".env" ]]; then
    ENV="--env-file .env"
else
    ENV=""
fi

if [[ -z "$TARGET_GPU" ]]; then
    echo "no target GPU set"
    echo "see nvidia-smi for available GPU's"
    echo "example: export TARGET_GPU=3"
    echo "for CPU use, set: export TARGET_GPU=none"
    exit
fi

if [[ -z "$CONTAINER_TAG" ]]; then
    echo "no container tag set"
    echo "example: export CONTAINER_TAG=mds-tensorflow:latest"
    exit
fi

if [[ -z "$DATA_VOLUME" ]]; then
    echo "no data volume set"
    echo "example: export DATA_VOLUME=/data/this_project/data"
    exit
fi

if [[ -z "$CODE_VOLUME" ]]; then
    echo "no code volume set"
    echo "example: export CODE_VOLUME=/data/this_project/code"
    exit
fi

if [[ "$TARGET_GPU"=="none" ]]; then
    sudo docker run -v $DATA_VOLUME:/data -v $CODE_VOLUME:/code $ENV -it $CONTAINER_TAG "${SCRIPT_TO_RUN}"
else
    sudo docker run --gpus "device=${TARGET_GPU}" -v $DATA_VOLUME:/data $ENV -v $CODE_VOLUME:/code -it $CONTAINER_TAG "${SCRIPT_TO_RUN}"
fi
