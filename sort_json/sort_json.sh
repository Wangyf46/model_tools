#!/usr/bin/env bash

#image_set=/home/huolu/data/data/HSJ/ADAS_Obj_v1_PL/ImageSets/Main/val.txt
#xmls_path=/home/huolu/data/data/HSJ/ADAS_Obj_v1_PL/Annotations/
#pics_path=/home/huolu/data/data/HSJ/ADAS_Obj_v1_PL/JPEGImages/
#save_path=/home/huolu/data/data/HSJ/ADAS_Obj_v1_PL/ADASv1_val.json
#dataset=CalmCar_ADAS


sort_path=/home/python/Documents/sort_test.json     #要筛选的json文件
save_path=/home/python/Documents/save_test.json     #筛选后保存的json文件
dataset=/home/python/Documents/sort_test            #要筛选的标签文件
script_path=$(cd "$(dirname "$0")"; pwd)
python3  $script_path/sort_json.py $sort_path $save_path $dataset

