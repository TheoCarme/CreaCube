import argparse
import os
import json

def make_parser():
    parser = argparse.ArgumentParser("split images")
    parser.add_argument(
        "-a",
        "--annotations_dir",
        type=str,
        default=r"..\YOLOX\datasets\COCO\annotations",
        help="The path to directory containing the instances_train2017.json and instances_val2017.json files.",
    )
    parser.add_argument(
        "-t",
        "--train_dir",
        type=str,
        default=r"..\YOLOX\datasets\COCO\train2017",
        help="The path to the train2017 directory aimed at receiving the train images.",
    )
    parser.add_argument(
        "-v",
        "--val_dir",
        type=str,
        default=r"..\YOLOX\datasets\COCO\val2017",
        help="The path to the val2017 directory aimed at receiving the train images.",
    )
    
    return parser



if __name__ == '__main__':

    args = make_parser().parse_args()
    annotations_dir = args.annotations_dir
    train_dir = args.train_dir
    val_dir = args.val_dir

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
