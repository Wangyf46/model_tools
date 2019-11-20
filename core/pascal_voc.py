from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import warnings
import os
from lxml import etree, objectify


def _get_check_element(root, name, length):
    """
     Acquire xml element, and check whether it is valid?
     :param root: root element
     :param name: element name
     :param length: nums of child element, if 0, return all child elements without check
     :return: element value, xml element
     """
    elements = root.findall(name)

    if 0 == len(elements):
        raise ValueError('Can not find %s in %s' % (name, root.tag))

    if len(elements) != length > 0:
        raise ValueError('The nums of %s is supposed to be %d, '
                         'but is %d' % (name, length, len(elements)))
    if 1 == length:
        elements = elements[0]

    return elements


def parse_xml(xml_path, image_ext='png'):
    """
    解析xml文件, 并返回解析后的结果
    对于单个XML文件，解析输出一个Annotaion的字典，格式如下
      Annotaion = {'image': {'file_name': xx, 'height': xx, 'width': xx},
                   'annotaion': [object_dict1, object_dict2, ...]}
      其中 object_dict1 = {'area': xx, 'bbox': [xmin, ymin, b_width, b_height], 'category': xxx}
    :param xml_path: xml文件的绝对路径
    :param image_ext: xml文件对应图片格式, 可以是'png', 'jpg'等
    :return:
    """

    annotation = {'image': None, 'annotation': []}

    xml_tree = etree.parse(xml_path)
    xml_root = xml_tree.getroot()

    name = os.path.splitext(os.path.basename(xml_path))[0]
    image_name = "%s.%s" % (name, image_ext)

    # parse image size
    size_element = _get_check_element(xml_root, 'size', 1)

    # 部分xml文件中的size保存的是float类型的数据,因此需要转换一下
    try:
        image_width = int(_get_check_element(size_element, 'width', 1).text)
        image_height = int(_get_check_element(size_element, 'height', 1).text)
    except ValueError:
        image_width = int(float(_get_check_element(size_element, 'width', 1).text))
        image_height = int(float(_get_check_element(size_element, 'height', 1).text))
    image_des = {'file_name': image_name, 'height': image_height, 'width': image_width}

    annotation['image'] = image_des

    # parse objects
    objs = xml_root.findall('object')

    for obj in objs:
        category = _get_check_element(obj, 'name', 1).text

        try:
            supercategory = _get_check_element(obj, 'supercategory', 1).text
        except ValueError:
            supercategory = 'none'

        bndbox = _get_check_element(obj, 'bndbox', 1)

        try:
            xmin = int(_get_check_element(bndbox, 'xmin', 1).text)
            ymin = int(_get_check_element(bndbox, 'ymin', 1).text)
            xmax = int(_get_check_element(bndbox, 'xmax', 1).text)
            ymax = int(_get_check_element(bndbox, 'ymax', 1).text)
        except ValueError:
            if _get_check_element(bndbox, 'xmin', 1).text=='NaN': #HL,数据处理错误
                continue
            xmin = int(float(_get_check_element(bndbox, 'xmin', 1).text))
            ymin = int(float(_get_check_element(bndbox, 'ymin', 1).text))
            xmax = int(float(_get_check_element(bndbox, 'xmax', 1).text))
            ymax = int(float(_get_check_element(bndbox, 'ymax', 1).text))


        #hl
#        score = float(_get_check_element(obj, 'score', 1).text)
        # if not (image_width - 1 > xmax > xmin >= 0 and image_height - 1 > ymax > ymin >= 0):
        #     raise RuntimeError("Find invalid object rect[%d, %d, %d, %d] in %s[%d, %d]"
        #                        % (xmin, ymin, xmax, ymax, image_name, image_width, image_height))

        # TODO: 当前的标注文件中存在xmax==image_width / ymax == image_width的情况
        if not (image_width >= xmax > xmin >= 0 and image_height >= ymax > ymin >= 0):
            warnings.warn("Find invalid object rect[%d, %d, %d, %d] in %s[%d, %d]"
                               % (xmin, ymin, xmax, ymax, image_name, image_width, image_height),
                          RuntimeWarning)

        obj_width = xmax - xmin + 1
        obj_height = ymax - ymin + 1
        obj_annotation = {'area': obj_width * obj_height,
                          'bbox': [xmin, ymin, obj_width, obj_height],
                          'category': category,
                          'supercategory': supercategory,
#                          'score': score,
                          'aspectratio': obj_width / obj_height,
                          }#hl->score新增  aspectratio宽高比

        annotation['annotation'].append(obj_annotation)

    return annotation


def dump_xml(annotaion, xml_path):
    """
    将annotation中的内容保存到xml文件
    :param annotaion:
    :param xml_path:
    :return:
    """
    # 构建xml的基础部分

    file_name = annotaion['image']['file_name']
    width = annotaion['image']['width']
    height = annotaion['image']['height']

    base_maker = objectify.ElementMaker(annotate=False)
    xml_tree = base_maker.annotation(
        base_maker.folder('CalmCar'),
        base_maker.filename(file_name),
        base_maker.source(
            base_maker.database('Unknown'),
        ),
        base_maker.size(
            base_maker.width(width),
            base_maker.height(height),
            base_maker.depth(3)
        ),
        base_maker.segmented(0),
    )

    # objects
    objs = annotaion['annotation']
    for obj in objs:
        category = obj['category']
        if 'supercategory' in obj.keys():
            super_category = obj['supercategory']
        else:
            super_category = 'none'
        xmin, ymin, width, height = obj['bbox']

        xmax = xmin + width - 1
        ymax = ymin + height - 1

        obj_maker = objectify.ElementMaker(annotate=False)
        obj_tree = obj_maker.object(
            obj_maker.name(category),
            obj_maker.supercategory(super_category),
            obj_maker.bndbox(
                obj_maker.xmin(xmin),
                obj_maker.ymin(ymin),
                obj_maker.xmax(xmax),
                obj_maker.ymax(ymax)
            ),
            obj_maker.difficult(0)
        )
        xml_tree.append(obj_tree)

    etree.ElementTree(xml_tree).Fwrite(xml_path, pretty_print=True)