#-*- coding: UTF-8 -*-
import os
import random
import shutil
import argparse
import math


def parse_args():
    parser = argparse.ArgumentParser('split Images dir and Annotation dir!')
    parser.add_argument('srcpath',
                        help='all files list.')
    parser.add_argument('dstpath',
                        help='Path including xml files and pictures files.')
    return parser.parse_args()


def get_path(args, img_files, xml_files):
    fs = os.listdir(args.srcpath)
    idx = 1
    for f in fs:
        tmp_path = os.path.join(args.srcpath, f)
        print(tmp_path,idx)
        idx = idx + 1
        if not os.path.isdir(tmp_path):
            if os.path.splitext(f)[1] == '.jpg'or os.path.splitext(f)[1] == '.jpeg' or os.path.splitext(f)[1] == '.png':
                img_files.append(os.path.splitext(f)[0])
                # img_path = args.dstpath + 'Images/'
                # if not os.path.exists(img_path):
                #     os.makedirs(img_path)
                # shutil.copy(tmp_path, img_path ) #拷贝文件
            elif os.path.splitext(f)[1] == '.xml':
                xml_files.append(os.path.splitext(f)[0])
                # xml_path =args.dstpath + 'Annotations/'
                # if not os.path.exists(xml_path):
                #     os.makedirs(xml_path)
                # shutil.copy(tmp_path, xml_path)
    print("has read all files")


def splitsample(args, img_files, xml_files):
    img_files_set = set(img_files)
    xml_files_set = set(xml_files)
    allfiles_set = img_files_set & xml_files_set
    allfiles_set = img_files_set
    allfiles = list(allfiles_set)           # set to list
    print(len(allfiles))
    test_file = random.sample(allfiles, math.ceil(len(allfiles)*0.2))  # 随机抽选
    test_file_set = set(test_file)
    trainval_file_set = allfiles_set-test_file_set
    val_file = random.sample(trainval_file_set, math.ceil(len(allfiles)*0.2))
    val_file_set=set(val_file)
    train_file_set = trainval_file_set-val_file_set
    with open(args.dstpath+'train.txt','w') as fw1:
        for name in train_file_set:
            fw1.write(args.dstpath + 'Annotations/' + name + '\n')
    with open(args.dstpath+'val.txt','w') as fw2:
        for name in val_file_set:
            fw2.write(args.dstpath + 'Annotations/' + name+'\n')
    with open(args.dstpath+'test.txt','w') as fw3:
        for name in test_file_set:
            fw3.write(args.dstpath + 'Annotations/' + name+'\n')


if __name__ == '__main__':
    img_files = []
    xml_files = []
    args = parse_args()
    get_path(args, img_files, xml_files)
    splitsample(args, img_files, xml_files)
    print('done')
