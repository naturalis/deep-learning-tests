#!/bin/bash

if [[ -z "$1" ]]; then
    echo "no script to run"
    exit
fi

sudo docker-compose run tensorflow "$1"