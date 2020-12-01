#!/bin/bash


PROJECT_ROOT=$(grep PROJECT_ROOT .env | cut -d '=' -f2)




# sudo docker-compose run  webserver /bin/sh -c 'rm /var/www/html/project'
# sudo docker-compose run  webserver /bin/sh -c 'ln -s $PROJECT_ROOT /var/www/html/project'

echo sudo docker-compose run  webserver /bin/sh -c 'ln -s $PROJECT_ROOT /var/www/html/project'
