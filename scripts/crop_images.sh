#!/bin/bash

INDIR=$1

if [ -z "$INDIR" ]; then
    echo "need a root image folder"
    exit
fi

for f in $(find ./ -type f -name '*.jpg');
do
    HEIGHT=$(convert $f -format "%h" info:)
    convert $f -crop 250x$HEIGHT+0+0 +repage $f
    echo $f
done

