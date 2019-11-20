#!/usr/bin/env bash

#image_set=/home/huolu/data/data/HSJ/ADAS_Obj_v1_360/ADASV1_360_train.json
#image_set=/home/yoershine/Data/calmcar_marker/instances_trainval2018.json
#data_path=/home/huolu/data/data/HSJ/ADAS_Obj_v1_360/Annotations
#save_path=./statistics
#fmt=coco

#image_set=/home/zhiyong/DATA/CalmCar_Object_2018/ImageSets/Main/train.txt
#image_set=/ssd-1t/dataset/CalmCar_Object_2018/annotations_v2/instances_train2018.json
#data_path=/home/zhiyong/DATA/CalmCar_Object_2018/Annotations
#save_path=./statistics
#fmt=pascal



#image_set=/home/huolu/data/data/FCM/object_dect/train.txt
#image_set=/ssd-1t/dataset/USA_LISA/instances_train2018.json
#data_path=/home/huolu/data/data/FCM/object_dect/Annotations
#save_path=/home/huolu/data/data/FCM/object_dect/statistics
#fmt=pascal

#image_set=/home/huolu/data/data/Fisheye/20190530/WJZ/train.txt
##image_set=/ssd-1t/dataset/USA_LISA/instances_train2018.json
#data_path=/home/huolu/data/data/Fisheye/20190530/WJZ/Annotations
#save_path=/home/huolu/data/data/Fisheye/20190530/WJZ/statistics



##image_set=/data/workspace/mixed-data/test.txt
#image_set=/data/workspace/mixed-data/test.json
#data_path=/data/workspace/mixed-data/Annotations
#save_path=/data/workspace/mixed-data/statistics
##fmt=pascal
#fmt=coco


##image_set=/data/workspace/mixed-data-crop/val.txt
#image_set=/data/workspace/mixed-data-crop/test.json
#data_path=/data/workspace/mixed-data-crop/Annotations
#save_path=/data/workspace/mixed-data-crop/statistics
##fmt=pascal
#fmt=coco


#image_set=/data/workspace/adas-TM-crop/train.txt
##image_set=/data/workspace/adas-TM-crop/train.json
#data_path=/data/workspace/adas-TM-crop/Annotations
#save_path=/data/workspace/adas-TM-crop/statistics
#fmt=pascal
##fmt=coco


#image_set=/data/workspace/mixed-data-2/val.txt
image_set=/data/workspace/mixed-data-2/test.json
data_path=/data/workspace/mixed-data-2/Annotations
save_path=/data/workspace/mixed-data-2/statistics
#fmt=pascal
fmt=coco

script_path=$(cd "$(dirname "$0")"; pwd)
python $script_path/../generate_data_dist.py $image_set --data-path $data_path --save-path $save_path --fmt $fmt
