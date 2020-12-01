#!/bin/bash


PROJECT_ROOT=$(grep PROJECT_ROOT .env | cut -d '=' -f2)

echo $PROJECT_ROOT


# sudo docker-compose run  webserver /bin/sh -c 'rm /var/www/html/project'
# sudo docker-compose run  webserver /bin/sh -c 'ln -s /data/museum/naturalis /var/www/html/project'
