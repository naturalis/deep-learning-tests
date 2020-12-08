#!/bin/bash

INDIR=$1

if [ -z "$INDIR" ]; then
    echo "need a root image folder"
    exit
fi

for f in $(find $INDIR -type f -name '*.jpg');
do
    DIM=$(convert "$f" -format "%hx%w" info:)
    HEIGHT=$(echo $DIM | sed 's/x[0-9]*$//')
    WIDTH=$(echo $DIM | sed 's/^[0-9]*x//')
    NEW_WIDTH=$(($WIDTH/2))
    convert "$f" -crop ${NEW_WIDTH}x${HEIGHT}+0+0 +repage "$f"
    echo "$f"
done

