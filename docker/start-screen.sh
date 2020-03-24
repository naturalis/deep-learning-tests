#!/bin/bash

LOGFILE=../../screen-logs/screen.log
SESSION_NAME=mds

if [[ ! -z "$1" ]]; then
    LOGFILE=$1
fi

if [[ ! -z "$2" ]]; then
    SESSION_NAME=$2
fi

screen -L -Logfile $LOGFILE -S $SESSION_NAME
