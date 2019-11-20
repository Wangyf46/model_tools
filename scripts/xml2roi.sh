#!/usr/bin/env bash

imgfile=/data/workspace/mixed-data/Images/
anno_path1=/data/workspace/mixed-data/Annotations/
output=/data/workspace/speed-limit-crop/

script_path=$(cd "$(dirname "$0")"; pwd)
python $script_path/../xml2roi.py $imgfile $anno_path1 $output
