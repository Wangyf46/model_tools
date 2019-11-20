#!/usr/bin/python

# pip install lxml

import sys
import os
import cv2
import json
import argparse
from tqdm import tqdm, trange
from core import parse_xml
import pdb


class Voc2Coco(object):
    def __init__(self, _args):
        self.m_image_set = _args.image_set
        self.m_xml_path = _args.xml_path
        self.m_pics_path = _args.pics_path
        self.m_image_ext = _args.image_ext if _args.image_ext is None else 'png'
        self.m_save_path = _args.save_path
        self.m_dataset = _args.dataset

        self.m_xml_index = None

        self.m_json_dict = {"images":[],
                            "type": "instances",
                            "annotations": [],
                            "categories": []}
        #self.m_categories = None
        self.m_categories = "Unsure_1"

        self._load_categories()
        self._load_xml_index()

    def _get_check_element(self, _root, _name, _length=None):
        """
        Acquire xml element, and check whether it is valid?
        :param _root: root element
        :param _name: element name
        :param _length: nums of child element
        :return: element value, xml element
        """
        elements = _root.findall(_name)

        if 0 == len(elements):
            raise ValueError('Can not find %s in %s' % (_name, _root.tag))

        if _length is not None:
            if _length > 0 and len(elements) != _length:
                raise ValueError('The nums of %s is supposed to be %d, '
                                 'but is %d' % (_name, _length, len(elements)))
            if 1 == _length:
                elements = elements[0]

        return elements

    def _load_categories(self):
        # pdb.set_trace()
        base_path = os.path.dirname(__file__)
        category_path = os.path.join(base_path, "dataset", "%s.label" % self.m_dataset)
        with open(category_path, 'r') as fcat:
            cats = fcat.readlines()
            cats = list(map(lambda x:x.strip(), cats))
            self.m_categories = dict(zip(cats, range(len(cats))))

    def _load_xml_index(self):
        with open(self.m_image_set, 'r') as fsets:
            self.m_xml_index = list(map(lambda x:x.strip(), fsets.readlines()))

    def convert(self):
        # 遍历所有的XML文件
        bnd_id = 1
        image_id = 1

        for xml_idx in tqdm(range(len(self.m_xml_index)), ncols=100, desc="VOC2COCO"):
            xml = self.m_xml_index[xml_idx]
            abs_xml_path = os.path.join(self.m_xml_path, "%s.xml" % xml)
            if not os.path.exists(abs_xml_path):
                raise ValueError("Non existed xml path: %s" % abs_xml_path)

            annotaion = parse_xml(abs_xml_path, "png")
            print(abs_xml_path+'\n')
            image_des = annotaion['image']
            image_des['id'] = image_id

            abs_image_path = os.path.join(self.m_pics_path, image_des['file_name'])
            if not os.path.exists(abs_image_path):
                tqdm.write("Non existed pic path: %s" % abs_image_path)
                continue

            self.m_json_dict['images'].append(image_des)

            # TODO: Support segmentation. Currently we do not support segmentation.
            objs = annotaion['annotation']
            if len(objs) == 0:
                tqdm.write(" There is no objects in xml : %s" % os.path.basename(xml))

            for obj in objs:

                # 所有标签均以小写格式保存，兼容xml中出现大写字母的情况
                category = obj['category'].lower()
                if category == 'prohibit':
                    category = 'limitspeed'

                # 排除不需要的标签
                if category not in self.m_categories:
                    continue


                '''
                if obj['area'] < 40*40:
                    print('area')
                    continue
                '''

                category_id = self.m_categories[category]
                obj_annotation = {'area': obj['area'], 'iscrowd': 0,
                                   'bbox': obj['bbox'],
                                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                                   'segmentation': [], 'image_id': image_id}
                self.m_json_dict['annotations'].append(obj_annotation)

                bnd_id += 1

            image_id += 1
        # category
        for category, category_id in self.m_categories.items():
            #ignore background
            if category == 'background':
                continue
            self.m_json_dict['categories'].append({'supercategory': category.split(' ')[0], 'id': category_id, 'name': category})

        # save json
        with open(self.m_save_path, 'w') as fjson:  ##todo

            json.dump(self.m_json_dict, fjson)


def parse_args():
    parser = argparse.ArgumentParser("Transform pascal format annotations into coco format!")
    parser.add_argument('image_set',
                        help='xml list file.')
    parser.add_argument('xml_path',
                        help='Path including xml files.')
    parser.add_argument('pics_path',
                        help='Path including pictures files.')
    parser.add_argument('save_path',
                        help='Path to save json annotation file.')
    parser.add_argument('dataset',
                        help='Must specify the dataset, '
                                        'and script will load responding label file automaticly.')
    parser.add_argument('--check-size', dest='check_size', help="Whether to check image size")
    parser.add_argument('--image-ext', dest='image_ext', help="Image's extention")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cvt = Voc2Coco(args)
    cvt.convert()
