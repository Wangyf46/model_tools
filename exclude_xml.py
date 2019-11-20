#!/usr/bin/python
# -*- coding=utf-8 -*-
import os
import os.path
import xml.dom.minidom
from xml.etree.ElementTree import ElementTree, Element
def read_xml(in_path):
    '''''读取并解析xml文件
       in_path: xml路径
       return: ElementTree'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree
 
def write_xml(tree, out_path):
    '''''将xml文件写出
       tree: xml树
       out_path: 写出路径'''
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
# ---------------search -----
 
def find_nodes(tree, path):
    '''''查找某个路径匹配的所有节点
       tree: xml树
       path: 节点路径'''
    return tree.findall(path)
if __name__ == "__main__":
    path ="/home/apt/Documents/五院/annotations"
    files = os.listdir(path)  # 得到文件夹下所有文件名称
    s = []
    for xmlFile in files:  # 遍历文件夹
        if not os.path.isdir(xmlFile):
            # 1. 读取xml文件
            # file = open("C:\\Users\\li.yang\\PycharmProjects\\untitled1\\file.txt")
    
            #for lines in file:
            #  refile = lines.replace('\n', '')  # 替换换行符
            #  tree=read_xml("C:\\Users\\li.yang\\PycharmProjects\\untitled1\\"+refile)
    
            tree = read_xml("/home/apt/Documents/五院/annotations/"+xmlFile)
            print("---------------------------")
            print(tree)
            # 4. 删除节点
            # 定位父节点
            print(xmlFile)
            nodes = find_nodes(tree,"object/name")
            objects=tree.findall('./object')
            i=0
            if (len(nodes) > 0):
                for objectNum in objects:
                        name = nodes[i].text
                        if(name=="person"):
                            del_nodes = tree._root
                            del_nodes.remove(objectNum)
                        i = i + 1
                            # 准确定位子节点并删除之
            # 6. 输出到结果文件
            write_xml(tree,"/home/apt/Documents/五院/annotations2/"+xmlFile)
