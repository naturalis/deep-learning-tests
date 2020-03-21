#!/bin/bash

LOGFILE=$1

if [[ -z "$LOGFILE" ]]; then
  LOGFILE=screen.log
fi

tail -f $LOGFILE | grep -v shuffle
