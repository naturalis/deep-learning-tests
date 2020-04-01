#!/bin/bash

LOGFILE=$1

if [[ -z "$LOGFILE" ]]; then
    LOGFILE=../../screen-logs/screen.log
fi

tail -n 50 -f $LOGFILE | grep -v shuffle
