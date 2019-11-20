#!/usr/bin/env bash

org_path=/ssd-1t/dataset/BSD/Annotations_ORG
image_path=/ssd-1t/dataset/BSD/JPEGIMages
save_path=/home/xzn/workspace/Tmp/Annotations/

script_path=$(cd "$(dirname "$0")"; pwd)
python $script_path/../validate_xmls.py $org_path $image_path $save_path