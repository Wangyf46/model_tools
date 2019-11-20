#!/usr/bin/env bash

#srcpath='/data/workspace/mixed-data/'
#dstpath='/data/workspace/'

#srcpath='/data/workspace/adas-TM-crop/Images/'
#dstpath='/data/workspace/adas-TM-crop/'

srcpath='/data/workspace/mixed-data-2/Images/'
dstpath='/data/workspace/mixed-data-2/'

script_path=$(cd "$(dirname "$0")"; pwd)
python $script_path/../split_img_xml.py $srcpath $dstpath