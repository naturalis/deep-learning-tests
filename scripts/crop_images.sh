#!/bin/bash

INDIR=$1

if [ -z "$INDIR" ]; then
    echo "need a root image folder"
    exit
fi

IFS=$'\n'
for f in $(find $INDIR -type f -name '*.jpg');
do
    mv $f ./tmp
    DIM=$(convert ./tmp -format "%hx%w" info:)
    HEIGHT=$(echo $DIM | sed 's/x[0-9]*$//')
    WIDTH=$(echo $DIM | sed 's/^[0-9]*x//')
    NEW_WIDTH=$(($WIDTH/2))
    convert ./tmp -crop ${NEW_WIDTH}x${HEIGHT}+0+0 +repage ./tmp
    mv ./tmp $f
    echo $f
done

