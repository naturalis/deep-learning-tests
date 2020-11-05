#!/bin/bash

INDIR=$1

if [ -z "$INDIR" ]; then
    echo "need a root image folder"
    exit
fi

for f in $(find ./ -type f -name '*.jpg');
do
    convert $f -resize 500x500 $f
    echo $f
done

