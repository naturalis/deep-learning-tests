#!/bin/bash

MSG=this

if [[ ! -z "$1" ]]; then
  MSG=$1
fi

sudo git add .
sudo git commit -m "$MSG"
sudo git push
