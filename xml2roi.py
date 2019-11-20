# -*- coding: UTF-8 -*-
#遍历某个文件夹下所有xml文件

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import glob
import cv2
import numpy as np
from core import parse_xml
import argparse
import mmcv
import xml.etree.cElementTree as ET
from xml.etree.cElementTree import ElementTree
import shutil
from tqdm import tqdm
from core import parse_xml, dump_xml

'''
功能:将xml中的目标有选择的裁剪出来,选择包括类别,大小,评分等
'''
def parse_args():
    parser = argparse.ArgumentParser(description='from xmls split obj')
    parser.add_argument('imgfile', type=str, help='imageset file, for example *.png')
    # parser.add_argument('image_set', type=str, help='imageset file, for example *.json, *.txt')
    parser.add_argument('anno_path1', help='Source path1 where xml locates')
    parser.add_argument('output', help='output limitspeed path')
    # parser.add_argument('result_path', help='result_path where roi locates')
    # parser.add_argument('save_path', help='Dst path where to save merged xml')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

#分析xml中所有类别和个数
def get_xml_category(objects,categorys):

    for obj in objects:
        cate = obj['category']
        if categorys.__contains__(cate):
            categorys[cate] += 1
        else:
            categorys.append(cate)
            categorys[cate] = 1


if __name__ == '__main__':
    args = parse_args()
    #args.image_set="/home/huolu/workspace/data/HSJ/image-xr-1-20190211/F20100101083859S/"
    # args.anno_path1="/home/huolu/workspace/data/HSJ/image-xr-1-20190211/"
    # args.result_path="/home/huolu/workspace/lbp/data/night/"

    #设置参数,通过调整参数控制或得的样本,
    fscore=0.01   #目标评分阈值限制
    #category=['car','bus','truck']
    #category_ned = ['misc'] #需要提取的类别
    category = ['LimitSpeed']
    aspectratio_min=0.9  #目标框宽高比限制
    aspectratio_max = 1.1  # 目标框宽高比限制
    area=32*32  #最小目标面积限制


    # 读取xml列表
    # xml_list = mmcv.list_from_file(args.image_set)
    # #map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回
    # xml_list = list(map(lambda x: x + '.xml', xml_list)) #lambda 创建一个匿名函数 map、reduce、filter、sorted等这些函数都支持函数作为参数，lambda函数就可以应用在函数式编程中

    #glob库:
    templist = glob.glob(args.anno_path1 + '/*.xml')
    xml_list_glob = []
    if len(templist) == 0:
        for file in glob.glob(args.anno_path1+'/*'):
            filelist = glob.glob(file + '/*.xml')
            xml_list_glob += filelist

    #f=os.listdir(path+'/')
    else:
        xml_list_glob=templist

    xml_list_glob.sort(key=lambda x: str(x[:-4]))
    mismatched_nums = 0
    print("xml_list_glob",len(xml_list_glob))

    # 遍历xml 列表
    flag=0
    num=0
    num_neg=0
    categorys = {}
    for i in tqdm(range(len(xml_list_glob)), ncols=100, desc='starting '):
        xml = xml_list_glob[i].split('/')[-1]
        abs_src_xml_path1 = xml_list_glob[i]

        abs_src_img_path1 =args.imgfile+os.path.splitext(xml)[0]+'.png'
        img = cv2.imread(abs_src_img_path1)  # xml和png不在同一个文件夹中

        #img = cv2.imread(os.path.splitext(abs_src_xml_path1)[0]+'.png')#xml和png在同一个文件夹中
        if img.data is None:
            print (f"{abs_src_img_path1}is not here")



        # cv2.imshow('a', img)
        # cv2.waitKey(2)
        #xml = xml_list[i]
        #abs_src_xml_path1 = os.path.join(args.anno_path1, xml)
        #save_path = os.path.join("/home/zhiyong/DATA/CalmCar_Object_2018/Annotations_5/", xml)

        #解析xml,写好的库
        annotation = parse_xml(abs_src_xml_path1)
        #print(len(annotation))
        objects = annotation["annotation"]

        #get_xml_category(objects,categorys)
        #input()

        #以下为负样本提取
        # if len(objects)==0:
        #     cv2.imwrite(args.result_path + os.path.splitext(xml)[0] + '_' + str(num_neg) + '.png', img)
        #     num_neg += 1
        #以下为正样本提取
        # for obj in  objects :
        #     #print(obj)
        #     #进行类别和其他限制调节判断
        #     if obj['category'] in category \
        #             and obj['area'] > area:
        #             #and obj['aspectratio'] > aspectratio_min and obj['aspectratio'] < aspectratio_max \
        #
        #        # and obj['score']>fscore \
        #        #     and obj['aspectratio'] < aspectratio \
        #        #     and obj['area'] > area: #and obj['aspectratio']<aspectratio \
        #         ixmin=obj['bbox'][0]
        #         ixmax = ixmin+obj['bbox'][2]
        #         iymin = obj['bbox'][1]
        #         iymax = iymin+obj['bbox'][3]
        #
        #         roi = np.zeros((obj['bbox'][3], obj['bbox'][2]), dtype=img.dtype)
        #         roi = img[iymin:iymax, ixmin:ixmax]
        #         #img[iymin:iymax, ixmin:ixmax] = roi
        #         cv2.imshow('b', roi)
        #         cv2.waitKey(2)
        #         #cv2.imwrite(args.result_path + os.path.splitext(xml)[0] + str(num)+'_'+str(obj['score']) +'_'+str(obj['bbox'][3])+ '.png', roi)
        #         cv2.imwrite(args.result_path + os.path.splitext(xml)[0] + str(num) + '_' + str(obj['bbox'][3]) + '.png', roi)
        #         num += 1
        # islimit=0
        itr = 0
        for obj in  objects :
            #print(obj)
            #进行类别和其他限制调节判断
            if obj['category'] in category :
                import pdb
                islimit=1
                # break

                x, y, h, w = obj['bbox']
                img_new = img.copy()
                # pdb.set_trace()
                # print(obj['bbox'])
                # cv2.imshow('src', img_new)
                # cv2.waitKey(0)

                # cv2.rectangle(img_new, (obj['bbox'][0], obj['bbox'][1]), (obj['bbox'][0] + obj['bbox'][2], obj['bbox'][1] + obj['bbox'][3]), (0, 255, 0), 1)
                # cv2.imshow('image', img_new)
                # cv2.waitKey(0)
                itr += 1
                img_limitspeed = img_new[obj['bbox'][1]:obj['bbox'][1]+obj['bbox'][3], obj['bbox'][0]:obj['bbox'][0]+obj['bbox'][2],]
                path = '/data/workspace/speed-limit-crop/all/' + str(itr) + os.path.splitext(xml)[0] + '.png'
                cv2.imwrite(path, img_limitspeed)
                # cv2.imshow('speed', img_limitspeed)
                # cv2.waitKey(0)

                # print(args.output + os.path.splitext(xml)[0] + '.png')
                # path_src=args.anno_path1+os.path.splitext(xml)[0]
                # path_dat=args.output + 'Annotations/' + os.path.splitext(xml)[0]
                # cv2.imwrite(args.output + 'Images/' + os.path.splitext(xml)[0] + '.png', img)
                # shutil.copyfile(path_src + '.xml', path_dat + '.xml')
        # if islimit==0:
        #     print(args.result_path + os.path.splitext(xml)[0]  + '.png')
        #     cv2.imwrite(args.result_path + os.path.splitext(xml)[0]+ '.png', img)

    print(f"categorys:{categorys}")

   # print("There are %d mismatched xmls in total!!!" % mismatched_nums)

