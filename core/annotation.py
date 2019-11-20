"""
Basic class used to descripe anntations, include bbox, segment
"""

from collections import namedtuple
import numpy as np

AnnoDes = namedtuple(
    'AnnoDes',
    [
        'area',             # int, object area
        'bbox',             # list or 1d numpy array, for example [xmin, ymin, width, height]
        'segmentation',     # list or 1d numpy array, for example [x1, y1, x2, y2 ...]
        'category',         # str, class name of object
        'super_category',   # str
        'is_crowd',         # 0 or 1
        'ignore',           # 0 or 1
    ]
)

ImageDes = namedtuple(
    'ImageDes',
    [
        'file_name',        # str, file_name with ext
        'height',           # int
        'width',            # int
    ]
)

Annotation = namedtuple(
    'Annotation',
    [
        'ImageDes',         # ImageDes
        'Annos',            # list of AnnoDes
    ]
)


def build_default_annodes(category, bbox, segments=None, super_category=None, is_crowd=0, ignore=0):

    if not isinstance(category, str):
        raise ValueError("category should be the class name of object")

    if isinstance(bbox, (list, np.ndarray)):
        if isinstance(bbox, np.ndarray):
            assert bbox.ndim == 1 and bbox.shape[0] == 4
        else:
            assert len(bbox) == 4
    else:
        raise ValueError("bbox should be list or ndarray")

    if segments is not None:
        assert isinstance(segments, (list, np.ndarray))
    else:
        segments = []

    super_category = super_category if super_category is not None else 'None'

    area = bbox[-1] * bbox[-2]

    return AnnoDes(area, bbox, segments, category, super_category, is_crowd, ignore)


def build_default_imgdes(file_name, height, width):
    assert isinstance(file_name, str)
    assert isinstance(height, int)
    assert isinstance(width, int)
    return ImageDes(file_name, height, width)


def build_default_annotation(img_des, annos=None):
    annos = [] if annos is None else annos
    return Annotation(img_des, annos)


