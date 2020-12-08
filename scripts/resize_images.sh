#!/bin/bash

INDIR=$1

if [ -z "$INDIR" ]; then
    echo "need a root image folder"
    exit
fi

IFS=$'\n'
for f in $(find $INDIR -type f -name '*.jpg');
do
    new_f=$(echo "$f" | sed 's/ /\\ /g')
    convert "$new_f" -resize 500x500^ "$new_f"
    echo "$new_f"
done

