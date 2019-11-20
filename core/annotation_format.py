"""
每张图片对应一个标注实例, 标注实例的格式如下
annotation_dict = {
    'image': {'file_name': xx, 'height': xx, 'width': xx },
    'annotation': [object_dict, object_dict, object_dict]
}
其中
object_dict = {
    'area': xx,
    'bbox': [xmin, ymin, b_width, b_height],
    'segmentation': [x1, y1, x2, y2, ...]
    'category': xxx,
    'supper_category': xxx
}
"""

