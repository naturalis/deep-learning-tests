#!/bin/bash

function get_env {
  VALUE=""
  IFS=$'\n'
  for ITEM in $(cat .env)
  do
    LINE=$(echo $ITEM | grep "^$1=")
    if [[ ! -z "$LINE" ]]; then
        VALUE=$(echo $LINE | sed -s 's/^[^=]*=//')
    fi
  done
  echo $VALUE
}

PROJECT_ROOT=$(get_env PROJECT_ROOT)

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "no PROJECT_ROOT found in .env"
    echo "exiting"
    exit
fi

echo "project root: ${PROJECT_ROOT}"

sudo docker-compose run  webserver /bin/sh -c "rm /var/www/html/project"
sudo docker-compose run  webserver /bin/sh -c "ln -s ${PROJECT_ROOT} /var/www/html/project"

