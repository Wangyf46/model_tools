#!/usr/bin/env bash

#image_set=/home/huolu/data/data/HSJ/ADAS_Obj_v1_PL/ImageSets/Main/val.txt
#xmls_path=/home/huolu/data/data/HSJ/ADAS_Obj_v1_PL/Annotations/
#pics_path=/home/huolu/data/data/HSJ/ADAS_Obj_v1_PL/JPEGImages/
#save_path=/home/huolu/data/data/HSJ/ADAS_Obj_v1_PL/ADASv1_val.json
#dataset=CalmCar_ADAS

#image_set=/home/huolu/data/data/PARKING/test.txt
#xmls_path=/home/huolu/data/data/PARKING/Annotations/
#pics_path=/home/huolu/data/data/PARKING/JPEGImages/
#save_path=/home/huolu/data/data/PARKING/Parking_test.json
#dataset=CalmCar_Parking


#image_set=/data/workspace/mixed-data/val.txt
#xmls_path=/data/workspace/mixed-data/Annotations/
#pics_path=/data/workspace/mixed-data/Images/
#save_path=/data/workspace/mixed-data/val.json
#dataset=mix_data

#image_set=/data/workspace/mixed-data-crop/train.txt
#xmls_path=/data/workspace/mixed-data-crop/Annotations/
#pics_path=/data/workspace/mixed-data-crop/Images/
#save_path=/data/workspace/mixed-data-crop/train.json
#dataset=mix_data

#image_set=/data/workspace/adas-TM-crop/val.txt
#xmls_path=/data/workspace/adas-TM-crop/Annotations/
#pics_path=/data/workspace/adas-TM-crop/Images/
#save_path=/data/workspace/adas-TM-crop/val.json
#dataset=mix_data


image_set=/data/workspace/mixed-data-2/train.txt
xmls_path=/data/workspace/mixed-data-2/Annotations/
pics_path=/data/workspace/mixed-data-2/Images/
save_path=/data/workspace/mixed-data-2/train.json
dataset=mix_data

script_path=$(cd "$(dirname "$0")"; pwd)
python $script_path/../voc2coco.py $image_set $xmls_path $pics_path $save_path $dataset
