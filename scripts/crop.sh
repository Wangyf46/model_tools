#!/usr/bin/env bash

imgpath=/data/workspace/adas-TM/Images/
xmlpath=/data/workspace/adas-TM/Annotations/
dstpath=/data/workspace/adas-TM-crop/

script_path=$(cd "$(dirname "$0")"; pwd)
python $script_path/../crop.py $imgpath  $xmlpath $dstpath
