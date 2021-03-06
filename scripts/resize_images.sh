#!/bin/bash

INDIR=$1

if [ -z "$INDIR" ]; then
    echo "need a root image folder"
    exit
fi

IFS=$'\n'
for f in $(find $INDIR -type f -name '*.jpg');
do
    # new_f=$(echo "$f" | sed 's/ /\\ /g')
    mv $f ./tmp
    # convert ./tmp -format "%hx%w" info:
    # echo
    convert ./tmp -resize 500x500^ ./tmp
    echo $f
    # convert ./tmp -format "%hx%w" info:
    # echo
    mv ./tmp $f
done

