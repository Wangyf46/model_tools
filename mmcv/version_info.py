import cv2
import torch


def use_opencv2():
    return cv2.__version__.split('.')[0] == '2'


def use_torchv1():
    return torch.__version__.split('.')[0] == '1'


USE_OPENCV2 = use_opencv2()
USE_TORCHV1 = use_torchv1()
