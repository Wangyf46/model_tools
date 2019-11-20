import sys
import os
import cv2
import argparse
from tqdm import tqdm
from core import parse_xml, dump_xml


def parse_args():
    parser = argparse.ArgumentParser("Validate xmls, for example add supercategory, remove illegal")
    parser.add_argument('org_path', help='Path including orignal xml files.')
    parser.add_argument('image_path', help='Path including orignal image files.')
    parser.add_argument('save_path', help='Path save validated xmls.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    src_path = args.org_path
    dst_path = args.save_path

    xmls = [item for item in os.listdir(src_path) if item.endswith('.xml')]

    for idx in tqdm(range(len(xmls)), ncols=100):
        xml_name = xmls[idx]

        abs_src_xml_path = os.path.join(src_path, xml_name)
        abs_dst_xml_path = os.path.join(dst_path, xml_name)
        abs_image_path = os.path.join(args.image_path, "%s.png" % os.path.splitext(xml_name)[0])
        annotation = parse_xml(abs_src_xml_path)

        image_des = annotation['image']

        if not os.path.exists(abs_image_path):
            tqdm.write("Non existed pic path: %s" % abs_image_path)
            continue

        cv_image = cv2.imread(abs_image_path)
        image_height, image_width= cv_image.shape[:2]

        if image_height != image_des['height'] or image_width != image_des['width']:
            image_des['height'] = image_height
            image_des['width'] = image_width

            tqdm.write("Invalid image width/height found in: %s" % xml_name)

        objs = annotation['annotation']
        b_valid = True
        validate_objs = []
        for obj in objs:
            # object 可能包含一些奇怪的名字, 比如 'car ' 'car \t'
            obj['category'] = obj['category'].strip()
            obj['supercategory'] = obj['category'].split(' ')[0]

            xmin, ymin, obj_width, obj_height = obj['bbox']
            xmax = xmin + obj_width - 1
            ymax = ymin + obj_height - 1
            if not (image_width >= xmax > xmin >= 0 and image_height >= ymax > ymin >= 0):
                b_valid = False

            validate_objs.append(obj)

        annotation['annotation'] = validate_objs
        annotation['image'] = image_des

        if b_valid:
            dump_xml(annotation, abs_dst_xml_path)
        else:
            tqdm.write("Invalid annotatins found in: %s" % xml_name)
