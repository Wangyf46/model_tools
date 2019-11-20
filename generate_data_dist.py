#!/usr/bin/env python
# encoding: utf-8

# ---------------------------------------------------------------------
# Author: CalmCar-yoershine
# Copyright: CalmCar 
# Name: generate_data_dist.py
# Create Time: 18-10-15 下午6:35
# Version: v0.0.1-beta
# Description:
#    statistics of detection datasets
# Changes:
# ---------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import itertools
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO

from core import parse_xml
'''生成样本分布
'''
def parse_args():
    """
    Parse input arguments
    :return: args
    """
    parser = argparse.ArgumentParser(description='statistics of pascal/coco style datasets')
    parser.add_argument('image_set', type=str, help='imageset file, for example *.json')
    parser.add_argument('--data-path', dest='data_path', type=str,
                        help='data path to datasets. If --fmt is pascal, --data-path must be specified')
    parser.add_argument('--save-path', dest='save_path', type=str,
                        help='If specified, statistics results and pie figure will be save in save-path')
    parser.add_argument('--fmt', dest='fmt', type=str, choices=['pascal', 'coco'],
                        default='pascal', help='which style? pascal or coco?')
    parser.add_argument('--spec-cats', dest='spec_cats', type=str,
                        default=None,
                        help='specify classes sperated by comma, default:car,bus,truck,person,cyclist,sign,misc')
    parser.add_argument('--areas', dest='areas', type=str,
                        default='32,96', help='specify area segments sperated by comma, default:"32,96", then scripts '
                                              'will count object nums responding with area')
    parser.add_argument('--use-diff', dest='use_diff', action='store_true', help='Include diffcult samples or not?')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def _init_area_segs(_args):
    """

    :param _args:
    :return: dict
    """
    seg_dict = dict()
    area_segs = list(map(lambda x: int(x.strip()), _args.areas.split(',')))
    for idx, area in enumerate(area_segs):
        if 0 == idx:
            seg_key = '(0*0,%d*%d]' % (area, area)
        else:
            seg_key = '(%d*%d,%d*%d]' % (area_segs[idx - 1], area_segs[idx - 1], area, area)

        if seg_key not in seg_dict.keys():
            seg_dict[seg_key] = 0

    # add last one
    seg_key = '(%d*%d, ]' % (area_segs[-1], area_segs[-1])

    if seg_key not in seg_dict.keys():
        seg_dict[seg_key] = 0

    return seg_dict


def _compute_area_segs(_args, _area, _seg_dict):
    """
    :param _args: input args
    :param _area: item area
    :param _seg_dict: existing dict
    :return: dict, for example {'(0-32*32]': 12, '(0-96*96]:10'}
    """
    area_segs = list(map(lambda x: int(x.strip()), _args.areas.split(',')))
    seg_key = None
    for idx, area in enumerate(area_segs):
        if _area <= area**2:
            if 0 == idx:
                seg_key = '(0*0,%d*%d]' % (area, area)
            else:
                seg_key = '(%d*%d,%d*%d]' % (area_segs[idx - 1], area_segs[idx - 1], area, area)
            break
    if seg_key is None :
        seg_key = '(%d*%d, ]' % (area_segs[-1], area_segs[-1])

    _seg_dict[seg_key] += 1

    return _seg_dict


def _get_check_element(_root, _name, _length):
    """
    Acquire xml element, and check whether it is valid?
    :param _root: root element
    :param _name: element name
    :param _length: nums of child element
    :return: element value, xml element
    """
    elements = _root.findall(_name)
    if 0 == len(elements):
        raise ValueError('Can not find %d in %s' % (_name, _root.tag))
    elif _length > 0 and len(elements) != _length:
        raise ValueError('The nums of %s is supposed to be %d, but is %d' % (_name, _length, len(elements)))

    if 1 == _length:
        elements = elements[0]

    return elements


def _load_pascal_annotations(_abs_xml_path, _args):
    """
    Parse annotations from XML file in Pascal VOC format
    :param _abs_xml_path: absolute path of xml file
    :param _args: input args
    :return: objects, type=dict, for example {'name': [class1, class2, ...], 'area': [area1, area2, ...]}
    """

    objects = dict()
    objects.setdefault('name', [])
    objects.setdefault('area', [])

    #annotaion = parse_xml(_abs_xml_path)
    #annotaion=dict()
    temp=parse_xml(_abs_xml_path)
    if temp is not None:
        annotaion=temp
    else:
        return None

    objs = annotaion['annotation']
    if len(objs) == 0:
        tqdm.write(" There is no objects in xml : %s" % os.path.basename(_abs_xml_path))

    for obj in objs:
        #hl->以下四行用来过滤标注不正确的xml文件
        # if obj['category'] == 'unknow' or obj['category'] == 'Copy of bus tail' or obj['category'] == 'Copy of car tail' \
        #  or obj['category'] == 'Copy of car head'  or obj['category'] == 'high_60' or obj['category'] == 'high_40':
        #     print(f"\n{_abs_xml_path} is {obj['category']}")
        #     return None

        objects['name'].append(obj['category'])
        objects['area'].append(obj['area'])

    return objects


def _pascal_statistics(_args):
    """

    :param _args:
    :return:
    """
    #hl
    deletename=[]
    detail_statistics = dict()

    with open(_args.image_set, 'r') as fset:
        xmls = ['%s.xml' % x.strip() for x in fset.readlines()]
    # Parse directory structures
    xml_path = args.data_path
    # jpeg_path = os.path.join(_args.data_path, 'JPEGImages')
    # xml_path = os.path.join(_args.data_path, 'Annotations')

    if not os.path.exists(xml_path):
        raise ValueError("Path not existed: %s" % xml_path)

    for id_xml in tqdm(range(len(xmls)), ncols=100, desc='Load pascal'):
        xml = xmls[id_xml]
        abs_xml_path = os.path.join(xml_path, xml)
        temp = _load_pascal_annotations(abs_xml_path, _args)
        print(temp)
        if temp is not None:
            objs=temp
            obj_names = objs['name']
            obj_areas = objs['area']

            for idx, obj_name in enumerate(obj_names):
                if obj_name not in detail_statistics.keys():
                    detail_statistics[obj_name] = _init_area_segs(_args)
                _compute_area_segs(args, obj_areas[idx], detail_statistics[obj_name])
        else :#记录错误标签的xml名字,
            deletename.append(xmls[id_xml])


    #print(len(deletename))
    #print(f"set(deletename)={set(deletename)}")
    #print(f"set(xml_path)={set(xml_path)}")
    file_set = set(xmls)-set(deletename)
    # with open('/home/huolu/workspace/data/calmcar_ADAS_V1/4689/new.txt','w') as fw:
    #     for name in file_set:
    #         fw.write(name.split(".")[0]+'\n')
    # sys.stdout.write('\n')
    return detail_statistics


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


# Subclass of COCO
class MyCoco(COCO):
    def __init__(self, annotation_file=None):
        super(MyCoco, self).__init__(annotation_file=annotation_file)

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else \
                [ann for ann in anns if areaRng[1] >= ann['area'] > areaRng[0]]

        if iscrowd is not None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids


def _coco_statistics(_abs_json_path, _args):
    """

    :param _abs_json_path: absolute path of json path
    :param _args: input args
    :return:
    """
    coco = MyCoco(_abs_json_path)

    total_image_ids = coco.getImgIds()
    print("***** Total images: %d *****" % len(total_image_ids))

    area_segs = list(map(lambda x: int(x.strip()), _args.areas.split(',')))

    detail_staticstics = dict()

    # Load coco dataset
    category_ids = coco.getCatIds()
    category_names = [c['name'] for c in coco.loadCats(category_ids)]
    cats = list(zip(category_ids, category_names))
    print(cats)
    for cat_id, cat_name in cats:
        detail_staticstics[cat_name] = dict()
        for idx, area in enumerate(area_segs):
            if 0 == idx:
                seg_key = '(0*0,%d*%d]' % (area, area)
                detail_staticstics[cat_name][seg_key] = len(coco.getAnnIds(catIds=cat_id, areaRng=[0, area**2]))
            else:
                seg_key = '(%d*%d,%d*%d]' % (area_segs[idx - 1], area_segs[idx - 1], area, area)
                detail_staticstics[cat_name][seg_key] = len(coco.getAnnIds(catIds=cat_id,
                                                                           areaRng=[area_segs[idx - 1]**2, area ** 2]))
        seg_key = '(%d*%d, ]' % (area_segs[-1], area_segs[-1])
        detail_staticstics[cat_name][seg_key] = len(coco.getAnnIds(catIds=cat_id,
                                                                   areaRng=[area_segs[-1] ** 2, float('inf')]))

    return detail_staticstics


if __name__ == '__main__':
    args = parse_args()

    print("Starting statics: \n"
          "\t Data path: %s\n"
          "\t Image set: %s\n"
          "\t Format: %s\n"
          "\t Specified catogries: %s\n"
          "\t Use diff: %s" % (args.data_path, args.image_set,
                               args.fmt, args.spec_cats, args.use_diff))

    assert os.path.isfile(args.image_set), "File argument supposed to be file: %s" % args.image_set

    if 'coco' == args.fmt:
        assert args.image_set.endswith('.json'), 'Input data path: %s is not json format'
        statistics = _coco_statistics(args.image_set, args)
    else:
        statistics = _pascal_statistics(args)

    # transpose
    statistics_df = pd.DataFrame(statistics).T
    # statistics_df = statistics_df.sort_index()
    df_shape = statistics_df.shape
    # compute total category counts
    statistics_df['count'] = statistics_df.apply(lambda x: x.sum(), axis=1)
    # compute total segs counts
    statistics_df.loc['total'] = statistics_df.apply(lambda x: x.sum(), axis=0)
    # calculate segs ratio
    statistics_df.loc['ratio'] = statistics_df.loc['total'][:df_shape[1]] / statistics_df.loc['total']['count']
    # calculate category ratio
    statistics_df['ratio'] = statistics_df['count'][:df_shape[0]] / statistics_df.loc['total']['count']

    # statistics_df = statistics_df.reset_index()
    #
    # columns_names = list(statistics_df.columns)
    # columns_names[0] = 'name'
    # statistics_df.columns = columns_names
    if args.spec_cats is not None:
        raise NotImplementedError

    print("******************************* Results: *******************************")
    print(statistics_df)

    # save results
    if args.save_path is not None:
        image_set_name = os.path.splitext(os.path.basename(args.image_set))[0]

        abs_csv_path = os.path.join(args.save_path, "%s.csv" % image_set_name)
        statistics_df.to_csv(abs_csv_path, index=None)
        # save figures
        # plot and save area seg ratios
        area_seg_ratio = statistics_df.loc['ratio'].dropna()
        names = list(area_seg_ratio.index)
        values = list(area_seg_ratio)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

        ax1.pie(values, labels=names, autopct='%2.2f%%')
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.set_title('Area Segments', fontsize=20)
        ax1.legend()

        statistics_df_nona = statistics_df.dropna()

        category_ratio = statistics_df_nona['ratio']
        names = list(category_ratio.index)
        values = list(category_ratio)

        ax2.pie(values, labels=names, autopct='%2.2f%%')
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax2.set_title('Category Distribution', fontsize=20)
        ax2.legend(loc='upper right')

        abs_pic_path = os.path.join(args.save_path, "%s.png" % image_set_name)
        plt.savefig(abs_pic_path)
