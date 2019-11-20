import sys
sys.path.append('.')

import shutil
import os
import pandas
from copy import deepcopy
from xml_about.xml_parser import parse_xml,dump_xml
from collections import Counter



if __name__ == '__main__':

    path = r"M:\LYL\glasses\camera1_assist"
    for parent, floders, files in os.walk(path):
        for file in files:

            if file.endswith("xml"):

                xml_path = os.path.join(path, file)
                print(xml_path.split("\\")[-2])
                ori_parse = parse_xml(xml_path)
                #print(ori_parse)
                ori_ann = ori_parse["annotation"]

                new_ann = list()

                for obj in ori_ann:
                    new_obj_ann = deepcopy(obj)
                    if new_obj_ann["name"] == "assist":
                        new_obj_ann["name"] = "Glasses"
                        new_ann.append(new_obj_ann)

                new_parse = deepcopy(ori_parse)
                new_parse["annotation"] = new_ann

                xml_save_path = xml_path.replace(xml_path.split("\\")[-2], "camera1_assistToGlasses")
                dump_xml(new_parse, xml_save_path)

            '''

                new_obj_ann = deepcopy(obj)
                new_bndbox = []
                for bnd in new_obj_ann["bndbox"]:
                    #print(bnd)
                    new_bndbox.append(round(bnd))
                #print(new_bndbox)
                new_obj_ann["bndbox"] = new_bndbox
                new_ann.append(new_obj_ann)

            new_parse = deepcopy(ori_parse)
            #print("modify前：",new_parse)

            new_parse["annotation"] = new_ann
            #print("modify后",new_parse)

            xml_save_path = xml_path.replace("First_4000", "round")
            dump_xml(new_parse, xml_save_path)
            '''


