# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:29:49 2022

@author: tcarme
"""

import os
import json

annotations_dir = r"C:\Users\tcarme\Documents\Stage_XLIM\YOLOX\datasets\cubes\annotations"
train_dir = r"C:\Users\tcarme\Documents\Stage_XLIM\YOLOX\datasets\cubes\train2017"
val_dir = r"C:\Users\tcarme\Documents\Stage_XLIM\YOLOX\datasets\cubes\val2017"

instances = set(())

train_file = open( (annotations_dir + '\\' + "instances_train2017.json"), "r")
val_file = open( (annotations_dir + '\\' + "instances_val2017.json"), "r")

train_json = json.load(train_file)
val_json = json.load(val_file)

train_imgs = set(())
for image in train_json["images"]:
    train_imgs.add(image["file_name"])
    
val_imgs = set(())
for image in val_json["images"]:
    val_imgs.add(image["file_name"])

os.chdir(train_dir)
for file_name in os.listdir() :
    if file_name in val_imgs :
        dst = val_dir + '\\' + file_name
        os.rename(file_name, dst)
    elif file_name not in train_imgs :
        os.remove(file_name)
