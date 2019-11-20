#-*- coding: UTF-8 -*-
import cv2
import os
import argparse
import shutil
import pdb
from copy import deepcopy


from xml_about.xml_parser import parse_xml,dump_xml



# def position_trans(input, k, b):
#     output = input * k + b
#     return output

def position_trans(input, Lower, Upper, threshold):
    if input <= Lower:
        input = 1
    elif input >= Upper:
        input = threshold - 1 -1
    else:
        input = input - Lower
    return input


def max(input1, input2):
    if input1 >= input2:
        return input1
    else:
        return input2

def min(input1, input2):
    if input1 < input2:
        return input1
    else:
        return input2


def crop(args):
    dst_img = os.path.join(args.dstpath, 'Images')
    dst_xml = os.path.join(args.dstpath, 'Annotations')

    ## create save dir
    if not os.path.isdir(dst_img):
        os.makedirs(dst_img)
    if not os.path.isdir(dst_xml):
        os.makedirs(dst_xml)

    idx = 0
    files_list = os.listdir(args.srcpath)
    for fname in files_list:
        name = os.path.splitext(fname)
        if name[1] == '.png' or name[1] == '.jpeg' or name[1] == '.png':
            src_img = cv2.imread(os.path.join(args.srcpath, fname))
            src_x0, src_y0 = (src_img.shape[0]-1)/2, (src_img.shape[1]-1)/2    ## TODO: center point
            dst_x0, dst_y0 = (args.height-1)/2, (args.width-1)/2
            x1, y1 = src_x0 - (args.height-1)/2, src_y0 - (args.width-1)/2
            x2, y2 = src_x0 + (args.height+1)/2, src_y0 + (args.width+1)/2

            ## transform coef-H
            k_h = (dst_x0 - 0) / (src_x0 - x1)
            b_h = -1.0 * k_h * x1

            ## tranform coef-W
            k_w = (dst_y0 - 0) / (src_y0 - y1)
            b_w = -1.0 * k_w * y1

            ## save crop img
            new_img = src_img[int(x1):int(x2), int(y1):int(y2)]
            cv2.imwrite(os.path.join(dst_img, fname), new_img)

            ## parse and save xml
            ori_parse = parse_xml(os.path.join(args.srcpath, name[0]) + '.xml')
            ori_ann = ori_parse["annotation"]
            new_ann = list()
            for obj in ori_ann:
                new_obj_ann = deepcopy(obj)

                ### TODO: Check Label Info

                ymin = max(new_obj_ann['bndbox'][1], x1)
                ymax = min(new_obj_ann['bndbox'][3], x2)
                xmin = max(new_obj_ann['bndbox'][0], y1)
                xmax = min(new_obj_ann['bndbox'][2], y2)
                if xmin > xmax or ymin > ymax:
                    # print('yuejie',fname)
                    new_obj_ann['bndbox'][0] = 0.0
                    new_obj_ann['bndbox'][1] = 0.0
                    new_obj_ann['bndbox'][2] = 0.0
                    new_obj_ann['bndbox'][3] = 0.0
                else:
                    ## bndbox area
                    ori_area = (new_obj_ann['bndbox'][2] - new_obj_ann['bndbox'][0]) * \
                               (new_obj_ann['bndbox'][3] - new_obj_ann['bndbox'][1])
                    new_area = (xmax - xmin) * (ymax-ymin)
                    if ori_area == 0:
                        # print('Label error',fname)
                        new_obj_ann['bndbox'][0] = 0.0
                        new_obj_ann['bndbox'][1] = 0.0
                        new_obj_ann['bndbox'][2] = 0.0
                        new_obj_ann['bndbox'][3] = 0.0
                    else:
                        if new_area / ori_area >= args.threshold:
                            # print(new_obj_ann['bndbox'])
                            new_obj_ann['bndbox'][0] = position_trans(new_obj_ann['bndbox'][0], y1, y2, args.width)
                            new_obj_ann['bndbox'][1] = position_trans(new_obj_ann['bndbox'][1], x1, x2, args.height)
                            new_obj_ann['bndbox'][2] = position_trans(new_obj_ann['bndbox'][2], y1, y2, args.width)
                            new_obj_ann['bndbox'][3] = position_trans(new_obj_ann['bndbox'][3], x1, x2, args.height)
                            # print(new_obj_ann['bndbox'])


                            '''
                            new_obj_ann['bndbox'][0] = position_trans(new_obj_ann['bndbox'][0], k_h, b_h)
                            if new_obj_ann['bndbox'][0] < 0.0:
                                new_obj_ann['bndbox'][0] = 0.0

                            new_obj_ann['bndbox'][1] = position_trans(new_obj_ann['bndbox'][1], k_w, b_w)
                            if new_obj_ann['bndbox'][1] < 0.0:
                                new_obj_ann['bndbox'][1] = 0.0

                            new_obj_ann['bndbox'][2] = position_trans(new_obj_ann['bndbox'][2], k_h, b_h)
                            if new_obj_ann['bndbox'][2] > args.height - 1:
                                new_obj_ann['bndbox'][2] = args.height - 1

                            new_obj_ann['bndbox'][3] = position_trans(new_obj_ann['bndbox'][3], k_w, b_w)
                            if new_obj_ann['bndbox'][3] > args.width - 1:
                                new_obj_ann['bndbox'][3] = args.width - 1
                            '''
                        else:
                            # print('area error',fname)
                            new_obj_ann['bndbox'][0] = 0.0
                            new_obj_ann['bndbox'][1] = 0.0
                            new_obj_ann['bndbox'][2] = 0.0
                            new_obj_ann['bndbox'][3] = 0.0
                '''
                ## position transform
                new_xmin = position_trans(new_obj_ann['bndbox'][0], k_h, b_h)
                new_xmin = np.where(new_xmin<0.0, 0.0, new_xmin)
                new_ymin = position_trans(new_obj_ann['bndbox'][1], k_w, b_w)
                new_ymin = np.where(new_ymin<0.0, 0.0, new_ymin)
                new_xmax = position_trans(new_obj_ann['bndbox'][2], k_h, b_h)
                new_xmax = np.where(new_xmax > args.height-1, args.height-1, new_xmax)
                new_ymax = position_trans(new_obj_ann['bndbox'][3], k_w, b_w)
                new_ymax = np.where(new_ymax>args.width-1, args.width-1, new_ymax)

                ## bndbox area
                ori_area = (new_obj_ann['bndbox'][2] - new_obj_ann['bndbox'][0] + 1) * \
                           (new_obj_ann['bndbox'][3] - new_obj_ann['bndbox'][1] + 1)
                area_new = (new_xmax - new_xmin + 1) * (new_ymax - new_ymin + 1)

                ## Check Error Label, eg.:2_2566/20100102_084516_000973.xml
                ### TODO: delete negative
                if ori_area == 0.0:
                    print(fname)
                    new_obj_ann['bndbox'][0] = 0.0
                    new_obj_ann['bndbox'][1] = 0.0
                    new_obj_ann['bndbox'][2] = 0.0
                    new_obj_ann['bndbox'][3] = 0.0
                else:
                    if area_new / ori_area >= args.threshold:
                        new_obj_ann['bndbox'][0] = new_xmin
                        new_obj_ann['bndbox'][1] = new_ymin
                        new_obj_ann['bndbox'][2] = new_xmax
                        new_obj_ann['bndbox'][3] = new_ymax
                    else:
                        new_obj_ann['bndbox'][0] = 0.0
                        new_obj_ann['bndbox'][1] = 0.0
                        new_obj_ann['bndbox'][2] = 0.0
                        new_obj_ann['bndbox'][3] = 0.0
               
                if new_obj_ann['bndbox'][0] < 0 or new_obj_ann['bndbox'][2] > (args.width-1) or \
                    new_obj_ann['bndbox'][1] < 0 or new_obj_ann['bndbox'][3] > (args.height-1):
                    pdb.set_trace()
                    print(fname)
                 '''
                new_ann.append(new_obj_ann)
            new_parse = deepcopy(ori_parse)
            new_parse["annotation"] = new_ann
            new_parse['image']['height'] = args.height
            new_parse['image']['width'] = args.width
            shutil.copy(os.path.join(args.srcpath, name[0]) + '.xml', dst_xml)
            dump_xml(new_parse, os.path.join(dst_xml, name[0]+'.xml'))
            idx = idx + 1
            print(idx)
    print('Completed Loading ')


def parse_args():
    parser = argparse.ArgumentParser('Crop image and Change xml')
    parser.add_argument('srcpath', help='Path incluing images and xmls')
    parser.add_argument('dstpath', help='Path to save croped images and corresponding xmls')
    parser.add_argument('--height', type=int, default=360,
                        help='Croped area height')
    parser.add_argument('--width', type=int, default=640,
                        help='Croped area width')
    parser.add_argument('--threshold', type=int, default=0.3,
                        help='Iou threshold')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    crop(args)
