#!/usr/bin/python

# pip install lxml

import sys
import os
import json
import argparse

class Sort_json (object):
    def __init__(self, _args):
        self.m_sort_path = _args.sort_path
        self.m_save_path = _args.save_path
        self.m_dataset = _args.dataset     #要筛选的标签文件
        self.m_json_dict = {"images":[],
                            "type": "instances",
                            "annotations": [],
                            "categories": []}
        self.m_categories = "Unsure_1"
        self.load_categories()
        self.sort()

    def load_categories(self):
        base_path = os.path.dirname(__file__)
        category_path = os.path.join(base_path, "dataset", "%s.label" % self.m_dataset)
        with open(category_path, 'r') as fcat:
            cats = fcat.readlines()
            cats = list(map(lambda x:x.strip(), cats))
            self.m_categories = dict(zip(cats, range(len(cats))))
    def sort(self):
        dataJson = json.load(open(self.m_sort_path, encoding='UTF-8'))
        for key in self.m_categories:  #x是要筛选的标签
            for i in range(len(dataJson['categories'])):
                if key == dataJson['categories'][i]['supercategory']:
                    for n in range(len(dataJson['annotations'])-1):
                        if dataJson['categories'][i]['id'] == dataJson['annotations'][n]['category_id']:
                            for a in range(len(dataJson['images'])-1):
                                if dataJson['annotations'][n]['image_id'] == dataJson['images'][a]['id'] and dataJson['images'][a] not in self.m_json_dict['images']:
                                    self.m_json_dict['images'].append(dataJson['images'][a])
                            self.m_json_dict['annotations'].append(dataJson['annotations'][n])
                    self.m_json_dict['categories'].append(dataJson['categories'][i])
        with open(self.m_save_path, 'w') as fjson:
            json.dump(self.m_json_dict, fjson)
def parse_args():
    parser = argparse.ArgumentParser("Sort json!")
    parser.add_argument('sort_path', help='Path source json file.')
    parser.add_argument('save_path', help='Path to save json annotation file.')
    parser.add_argument('dataset', help='Must specify the dataset, '
                                        'and script will load responding label file automaticly.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()
if __name__ == '__main__':

    args = parse_args()
    cvt = Sort_json(_args=args)
    #cvt.sort()
