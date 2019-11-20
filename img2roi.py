#!/usr/bin/env python


import cv2
global img
global point1, point2


def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 5, (255, 0, 0), 1)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (0, 255, 0), 1)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 1)
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        cut_img = img[min_y:min_y + height, min_x:min_x + width]
        path = '/data/workspace/speed-limit-crop/' + 'adas-TM-' + fname
        cv2.imwrite(path, cut_img)


def main():
    import os
    global img
    global fname
    # dir = '/data/workspace/mixed-data/Images/'
    dir = '/data/workspace/adas-TM/Images/'
    files = os.listdir(dir)
    files.sort(key=lambda x: str(x[:-4]))
    itr = 1
    for fname in files:
        print(fname, itr)
        path = os.path.join(dir, fname)
        img = cv2.imread(path)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_mouse)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        itr += 1


if __name__ == '__main__':
    main()
