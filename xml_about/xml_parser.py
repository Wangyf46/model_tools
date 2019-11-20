from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import os
import uuid as ud
from lxml import etree

"""
每张图片对应一个标注实例, 标注实例的格式如下
annotation_dict = {
    'image': {'file_name': xx, 'height': xx, 'width': xx },
    'annotation': [object_dict, object_dict, object_dict]
}
其中
object_dict = {
    'bndbox': [xmin, ymin, xmax, y_max], rectangle 必选, float
    'points': [[x1, y1], [x2, y2], ...], polygon 必选, float
    'type': 'rectangle' / 'polygon', 可选
    'uuid': '3dbf02e8-5a19-44e5-b0de-65155ee8fbac', 可选
    'attributes': dict{xx:xx, xx:xx, xx:xx}, 可选
    'pose': xx, 未使用
    'truncated': xx, 未使用
    'difficult': 0 / 1, 未使用
    'score: 0.75, 可选
    'name': 'car',
    'worker': xxx, 标注人员
}
"""


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
        return etree.Element('default')

    if len(elements) != length:
        raise RuntimeError('The nums of %s is supposed to be %d, '
                           'but is %d' % (name, length, len(elements)))
    if 1 == length:
        elements = elements[0]

    return elements

def parse_xml(xml_path, image_ext = "png"):
    """
    解析xml文件, 并返回解析后的结果
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
    image_width = float(_get_check_element(size_element, 'width', 1).text)
    image_height = float(_get_check_element(size_element, 'height', 1).text)

    # 获取标注人员
    worker = _get_check_element(xml_root, 'worker', 1).text

    annotation['image'] = dict(
        file_name=image_name,
        height=image_height,
        width=image_width,
        worker=worker,
    )
    # import pdb
    # pdb.set_trace()
    # parse objects
    objs = xml_root.findall('object')

    for obj in objs:
        # 获取标注格式, 即 rectangle / polygon
        ann_type = _get_check_element(obj, 'type', 1).text

        # 获取uuid
        uuid = _get_check_element(obj, 'uuid', 1).text

        # 获取目标名字
        name = _get_check_element(obj, 'name', 1).text

        # pose
        pose = _get_check_element(obj, 'pose', 1).text
        # trucated
        truncated = _get_check_element(obj, 'truncated', 1).text
        # difficult
        difficult = _get_check_element(obj, 'difficult', 1).text

        # 获取详细标注
        numeric_points = None
        numeric_bndbox = None

        if ann_type is None or 'rectangle' == ann_type:
            bndbox_node = _get_check_element(obj, 'bndbox', 1)
            xmin = float(_get_check_element(bndbox_node, 'xmin', 1).text)
            ymin = float(_get_check_element(bndbox_node, 'ymin', 1).text)
            xmax = float(_get_check_element(bndbox_node, 'xmax', 1).text)
            ymax = float(_get_check_element(bndbox_node, 'ymax', 1).text)
            numeric_bndbox = [xmin, ymin, xmax, ymax]
        elif ann_type == 'polyline': #or ann_type == "polyline":
            points_node = _get_check_element(obj, 'points', 1)
            numeric_points = []
            sub_points_node = points_node.findall('point')
            for sub_point_node in sub_points_node:
                x = float(_get_check_element(sub_point_node, 'x', 1).text)
                y = float(_get_check_element(sub_point_node, 'y', 1).text)
                numeric_points.append([x, y])
        else:
            raise RuntimeError("Unspported ann type: %s" % ann_type)

        # 读取属性
        attrs_node = _get_check_element(obj, 'attributes', 1)
        if len(attrs_node) == 0:
            attrs_dict = None
        else:
            attrs_dict = dict()
            for attr_node in attrs_node:
                attr_tag = attr_node.tag
                attr_value = attr_node.text
                attrs_dict[attr_tag] = attr_value

        obj_annotation = dict(
            type=ann_type,
            uuid=uuid,
            name=name,
            pose=pose,
            truncated=truncated,
            difficult=difficult,
            bndbox=numeric_bndbox,
            points=numeric_points,
            attributes=attrs_dict
        )
        annotation['annotation'].append(obj_annotation)

    return annotation


def dump_xml(annotaion, xml_path):
    """
    将annotation中的内容保存到xml文件
    :param annotaion:
    :param xml_path:
    :return:
    """
    file_name = annotaion['image']['file_name']
    width = annotaion['image']['width']
    height = annotaion['image']['height']
    worker = annotaion['image']['worker']

    root = etree.Element("annotation")

    etree.SubElement(root, "worker").text = 'Unknow' if worker is None else worker
    etree.SubElement(root, "folder").text = "CalmCar"
    etree.SubElement(root, "filename").text = file_name

    source_node = etree.SubElement(root, "source")
    etree.SubElement(source_node, "database").text = "Unknown"

    size_node = etree.SubElement(root, "size")
    etree.SubElement(size_node, "width").text = str(width)
    etree.SubElement(size_node, "height").text = str(height)
    etree.SubElement(size_node, "depth").text = str(3)

    etree.SubElement(root, 'segmented').text = str(0)

    # objects
    objs = annotaion['annotation']
    for obj in objs:
        name = obj['name']
        uuid = obj['uuid'] if obj['uuid'] is not None else ud.uuid1()
        ann_type = obj['type']
        pose = obj['pose']
        truncated = obj['truncated']
        difficult = obj['difficult']
        attributes = obj['attributes']
        bndbox = obj['bndbox']
        points = obj['points']

        # 如果没有指定标注类型, 比如老的标注文件, 那么根据bndbox和point是否为None来确定类型
        if ann_type is None:
            if bndbox is None and points is not None:
                ann_type = "polygon"
            elif points is None and bndbox is not None:
                ann_type = 'rectangle'
            else:
                raise RuntimeError("Unsupported type")

        obj_node = etree.SubElement(root, 'object')
        etree.SubElement(obj_node, 'name').text = name
        etree.SubElement(obj_node, 'type').text = ann_type
        etree.SubElement(obj_node, 'uuid').text = str(uuid)

        attr_node = etree.SubElement(obj_node, 'attributes')
        if attributes is not None:
            for tag, value in attributes.items():
                sub_attr_node = etree.SubElement(attr_node, tag)
                sub_attr_node.text = str(value)

        etree.SubElement(obj_node, 'pose').text = 'Unspecified' if pose is None else str(pose)
        etree.SubElement(obj_node, 'truncated').text = str(0) if truncated is None else str(truncated)
        etree.SubElement(obj_node, 'difficult').text = str(0) if difficult is None else str(difficult)

        if 'rectangle' == ann_type:
            bndbox_node = etree.SubElement(obj_node, 'bndbox')
            etree.SubElement(bndbox_node, 'xmin').text = str(bndbox[0])
            etree.SubElement(bndbox_node, 'ymin').text = str(bndbox[1])
            etree.SubElement(bndbox_node, 'xmax').text = str(bndbox[2])
            etree.SubElement(bndbox_node, 'ymax').text = str(bndbox[3])

        elif 'polygon' == ann_type:
            points_node = etree.SubElement(obj_node, 'points')
            for sub_points in points:
                sub_points_node = etree.SubElement(points_node, 'point')
                etree.SubElement(sub_points_node, 'x').text = str(sub_points[0])
                etree.SubElement(sub_points_node, 'y').text = str(sub_points[1])
    doc = etree.tostring(root, encoding='UTF-8', pretty_print=True, xml_declaration=True)
    with open(xml_path, 'wb') as fxml:
        fxml.write(doc)
