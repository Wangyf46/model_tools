import sys
import os
import cv2
import argparse
from tqdm import tqdm
from core import parse_xml, dump_xml


def calmcar_object(category):
    if category not in ['cycle', 'cyclist', 'tricycle', 'barricade']:
        return category.split(' ')[0]
    elif category in ['cycle', 'cyclist', 'tricycle']:
        return 'cyclist'
    elif category in ['barricade']:
        return 'misc'
    else:
        return category


def parse_args():
    parser = argparse.ArgumentParser("merge/remove/tranfer category")
    parser.add_argument('org_path', help='Path including orignal xml files.')
    parser.add_argument('save_path', help='Path save validated xmls.')
    parser.add_argument('--dataset', dest='dataset', choices=['object', 'tsr'], default='object', help='')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    base_path = os.path.dirname(__file__)

    transfer_func = None
    if args.dataset == 'object':
        transfer_func = calmcar_object

    src_path = args.org_path
    dst_path = args.save_path

    xmls = [item for item in os.listdir(src_path) if item.endswith('.xml')]

    for idx in tqdm(range(len(xmls)), ncols=100):
        xml_name = xmls[idx]

        abs_src_xml_path = os.path.join(src_path, xml_name)
        abs_dst_xml_path = os.path.join(dst_path, xml_name)

        annotation = parse_xml(abs_src_xml_path)
        image_des = annotation['image']

        objs = annotation['annotation']

        validate_objs = []
        for obj in objs:
            obj['category'] = transfer_func(obj['category'])
            validate_objs.append(obj)

        annotation['annotation'] = validate_objs
        annotation['image'] = image_des

        dump_xml(annotation, abs_dst_xml_path)
