import albumentations as A
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse



def make_parser():
    parser = argparse.ArgumentParser("change images dim")
    parser.add_argument(
        "-ii",
        "--input_images",
        type=str,
        help="The path to directory containing the source images.",
    )
    parser.add_argument(
        "-ij",
        "--input_json",
        type=str,
        help="The path to the source annotations.json file.",
    )
    parser.add_argument(
        "-oi",
        "--output_images",
        type=str,
        default="Converted_Images",
        help="The path of the directory to where the images converted to the specified dimensions should be saved.",
    )
    parser.add_argument(
        "-oj",
        "--output_json",
        type=str,
        default="converted_annotations.json"
        help="The path to where the json with the updated dimension should be saved.",
    )
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        default=640
        help="The value of the horizontal dimension to which the images should be converted. Make sure to chosse a value compatible to the 16/9 format",
    )
    
    return parser
    
    

def json_annotation_converter(json_path, converted_json_path, image_folder_path, image_save_path,size=(512,512)):
    image_folder_path = image_folder_path
    image_save_path = image_save_path
    images = glob(image_folder_path+"\\*.jpeg")
    H,W = size
    red = [0,255,0]

    json_path = json_path

    with open(json_path) as json_file:
        json_data = json.loads(json_file.read())
        json_file.close()

    list_keys =list(json_data.keys())
    new_json = json_data.copy()
    transform = A.Compose([
        A.Resize(height =H,width=W)
    ], keypoint_params=A.KeypointParams(format='xy'))


    for iidx in range(len(images)):
        list_xy = []
        key_jsondata = list_keys[iidx]
        data = json_data[key_jsondata]
        file_name = data["filename"]
        image = cv2.imread(image_folder_path + file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for obj in range(len(data["regions"])):
            print(obj)
            x = data["regions"][obj]["shape_attributes"]["all_points_x"]
            y = data["regions"][obj]["shape_attributes"]["all_points_y"]

            for i in range(len(x)):
                list_xy.append((x[i],y[i]))

        keypoints = list_xy
        transformed = transform(image = image, keypoints=keypoints)
        trans_image = transformed["image"]
        trans_image = cv2.cvtColor(trans_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_save_path+file_name,trans_image)
        
        s = 0
        for obj in range(len(new_json[key_jsondata]["regions"])):
            e = len(new_json[key_jsondata]["regions"][obj]["shape_attributes"]["all_points_x"]) + s
            new_xy = transformed["keypoints"][s:e]
            print("kp to json", new_xy)
            x = []
            y = []
            for idx in range(len(new_xy)):
                x.append(int(new_xy[idx][0]))
                y.append(int(new_xy[idx][1]))
        
            new_json[key_jsondata]["regions"][obj]["shape_attributes"]["all_points_x"] = x
            new_json[key_jsondata]["regions"][obj]["shape_attributes"]["all_points_y"] = y
            s = e
            
        with open(converted_json_path, 'w') as fp:
            json.dump(new_json, fp)

if __name__ == '__main__':

    args = make_parser().parse_args()

    image_folder_path = args.input_images
    json_path = args.input_json
    converted_json_path = args.output_json
    image_save_path = args.output_images
    size = (args.dimension, int(args.dimension/(16/9)))
                    
    json_annotation_converter(json_path,converted_json_path,image_folder_path, image_save_path, size = size)
