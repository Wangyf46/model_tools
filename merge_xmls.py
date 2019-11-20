from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import os
import sys
import mmcv
from tqdm import tqdm
from core import parse_xml, dump_xml


def parse_args():
    parser = argparse.ArgumentParser(description='merge xmls')
    parser.add_argument('image_set', type=str, help='imageset file, for example *.json, *.txt')
    parser.add_argument('anno_path1', help='Source path1 where xml locates')
    parser.add_argument('anno_path2', help='Source path2 where xml locates')
    parser.add_argument('save_path', help='Dst path where to save merged xml')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 读取xml列表
    xml_list = mmcv.list_from_file(args.image_set)
    xml_list = list(map(lambda x: x + '.xml', xml_list))

    mismatched_nums = 0
    # 遍历xml 列表
    for i in tqdm(range(len(xml_list)), ncols=100, desc='Merging '):
        xml = xml_list[i]
        abs_src_xml_path1 = os.path.join(args.anno_path1, xml)
        abs_src_xml_path2 = os.path.join(args.anno_path2, xml)

        if not (os.path.exists(abs_src_xml_path1) and os.path.exists(abs_src_xml_path2)):
            mismatched_nums += 1
            tqdm.write("**** Mismatched xml: %s" % xml)
            continue

        src_anno1 = parse_xml(abs_src_xml_path1)
        src_anno2 = parse_xml(abs_src_xml_path2)

        # # check whether image is same
        # assert src_anno1['image']['file_name'] == src_anno2['image']['file_name'] \
        #        and src_anno1['image']['width'] == src_anno2['image']['width'] \
        #        and src_anno1['image']['height'] == src_anno2['image']['height'], "Mismatched xml %s" % xml

        dst_anno = src_anno1.copy()
        dst_anno['annotation'] += src_anno2['annotation']

        abs_dst_xml_path = os.path.join(args.save_path, xml)
        dump_xml(dst_anno, abs_dst_xml_path)

    print("There are %d mismatched xmls in total!!!" % mismatched_nums)