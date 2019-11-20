#!/usr/bin/env bash

org_path=/ssd-1t/dataset/BSD/Annotations_11
save_path=/home/xzn/workspace/Tmp/Annotations/
dataset='object'

script_path=$(cd "$(dirname "$0")"; pwd)
python $script_path/../transfer_category.py $org_path $save_path  --dataset $dataset