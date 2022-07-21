# Created on Mon May 23 09:56:48 2022
# @author: Théo Carme : theo.carme63@gmail.com
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# -*- coding: utf-8 -*-

import argparse
import os
from alive_progress import alive_bar
import csv

import cv2 as cv
import numpy as np

from yolox.utils import mkdir

from cubes_detector import Cubes_Detector
from hands_tracker import Hands_Tracker


def make_parser():
    parser = argparse.ArgumentParser("combined detection")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=r"C:\Users\theon\Documents\Stage_XLIM\Livrable\YOLOX\ONNX_Models\yolox_s.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-v",
        "--video_path",
        type=str,
        default=r"F:\Videos\p659.mp4",
        help="Path to your input video.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=r"F:\Results",
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "-S",
        "--start",
        type=int,
        default=0,
        help="Number of seconds to ignore at the start of video.",
    )
    parser.add_argument(
        "-E",
        "--end",
        type=int,
        default=0,
        help="Number of seconds to ignore at the end of video.",
    )
    parser.add_argument(
        "-n",
        "--num_hands",
        type=int,
        default=4,
        help="Number of hands to track on the video.",
    )
    parser.add_argument(
        "-c",
        "--csv",
        action="store_true",
        help="Whether you want to create csv trace file.",
    )
    parser.add_argument(
        "-w",
        "--write_video",
        action="store_true",
        help="Whether you want to write the video.",
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="Whether you want to display the processed frames.",
    )
    
    return parser



def draw_bounding_box(event, x, y, flags, params):
    global counter
    global point_matrix
    # Left button mouse click event opencv
    if event == cv.EVENT_LBUTTONDOWN:
        point_matrix[counter] = [x,y]
        counter = counter + 1



def get_drawn_bounding_box(origin_img, colour) :    
    global point_matrix
    global counter
    
    window_name = f"[FIRST FRAME] Please draw the bounding box of the {colour} cube then press 'v' to validate it or 'r' to restart."
    img = np.copy(origin_img)
    point_matrix = np.zeros((2,2), int)
    counter = 0
    bounding_box = None
    
    print(window_name)
    while True :
    
        for j in range(2) :
            cv.circle(img, (point_matrix[j][0], point_matrix[j][1]), 3, (0, 255, 0), cv.FILLED)
    
        if counter == 2:
            x1 = point_matrix[0][0]
            y1 = point_matrix[0][1]
    
            x2 = point_matrix[1][0]
            y2 = point_matrix[1][1]
            # Draw rectangle for area of interest
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            bounding_box = np.array([x1, y1, x2, y2], dtype=int)
    
        # Showing original image
        cv.imshow(window_name, img)
        # Mouse click event on original image
        cv.setMouseCallback(window_name, draw_bounding_box, img)
        # Refreshing window all time
        key = cv.waitKey(1)
        if key == ord('v') :
            break
        elif key == ord('r') :
            counter = 0
            point_matrix = np.zeros((2,2), int)
            img = np.copy(origin_img)
                
    cv.destroyWindow(window_name)
    
    return bounding_box
    

#--------------------------------------------------------
#
#   ||\\      //||        //\\        ()   ||\\     ||
#   || \\    // ||       //  \\            || \\    ||
#   ||  \\  //  ||      //    \\      ||   ||  \\   ||
#   ||   \\//   ||     //======\\     ||   ||   \\  ||
#   ||          ||    //        \\    ||   ||    \\ ||
#   ||          ||   //          \\   ||   ||     \\||
#
#--------------------------------------------------------
    
    
if __name__ == '__main__':

    # Collect the parameters given as arguments
    args = make_parser().parse_args()

    # Get the input shape for the cube detector
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Open the video whose name was given as argument
    vid = cv.VideoCapture(args.video_path)
    # Get the resolution, the number of frames per second and the total number of frames of the video
    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    fps = vid.get(cv.CAP_PROP_FPS)
    nb_frames = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    
    # Calculate the number of frames to ignore at the beginining of the video, based on the given number of seconds to ignore.
    start = int(args.start*fps)
    end = int(args.end*fps)    
    # Get the maximum number of hands to track on the video
    num_hands = int(args.num_hands)
    # Get the information whether the csv file should be created, whether the video with drawn labels should be saved and whether the processed frames of the video should be displayed or not during execution
    csv_write = args.csv
    display = args.display
    write_video = args.write_video
        
    # If asked prepare the output video file to receive the processed frames.
    if write_video :
        # Assemble the path of the output directory, create it in the given general output directory name and the name of the input video. This directory is intended to store the output video and the csv file.
        output_dir = os.path.join(args.output_dir, os.path.basename(args.video_path).split('.')[0])
        mkdir(output_dir)
        # Assemble the path of the otput video and create the corresponding file.
        output_path = os.path.join(output_dir, os.path.basename(args.video_path).split('.')[0]) + ".avi"
        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        # Give the output video the same resolution and frame frequency as the input video.
        vid_out = cv.VideoWriter(output_path, fourcc, fps, (width,  height))
        
    # Drop as much frames as it was asked.
    for k in range(start) :
        vid.read()
    
    # If asked prepare the csv file to receive the data.
    if csv_write :
        # Assemble the path of the output directory, create it in the given general output directory name and the name of the input video. This directory is intended to store the output video and the csv file.
        output_dir = os.path.join(args.output_dir, os.path.basename(args.video_path).split('.')[0])
        mkdir(output_dir)
        # Assemble the path of the csv file.
        csv_path = os.path.join(output_dir, os.path.basename(args.video_path).split('.')[0]) + ".csv"
        # Create and open in 'write' mode the csv file.
        table_results = open(csv_path, mode = 'w', newline = '')    
        table_writer = csv.writer(table_results, delimiter = ',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
   
        # Assemble the header row to be written in the csv file.
        cube_coords = ["x1", "y1", "x2", "y2", "xc", "yc", "Score de confiance"]
        hand_coords = ["latéralité", "x wrist", "y wrist", "z wrist",\
                       "x thumb_cmc", "y thumb_cmc", "z thumb_cmc",\
                       "x thumb_mcp", "y thumb_mcp", "z thumb_mcp",\
                       "x thumb_ip", "y thumb_ip", "z thumb_ip",\
                       "x thumb_tip", "y thumb_tip", "z thumb_tip",\
                       "x index_mcp", "y index_mcp", "z index_mcp",\
                       "x index_pip", "y index_pip", "z index_pip",\
                       "x index_dip", "y index_dip", "z index_dip",\
                       "x index_tip", "y index_tip", "z index_tip",\
                       "x middle_mcp", "y middle_mcp", "z middle_mcp",\
                       "x middle_pip", "y middle_pip", "z middle_pip",\
                       "x middle_dip", "y middle_dip", "z middle_dip",\
                       "x middle_tip", "y middle_tip", "z middle_tip",\
                       "x ring_mcp", "y ring_mcp", "z ring_mcp",\
                       "x ring_pip", "y ring_pip", "z ring_pip",\
                       "x ring_dip", "y ring_dip", "z ring_dip",\
                       "x ring_tip", "y ring_tip", "z ring_tip",\
                       "x pinky_mcp", "y pinky_mcp", "z pinky_mcp",\
                       "x pinky_pip", "y pinky_pip", "z pinky_pip",\
                       "x pinky_dip", "y pinky_dip", "z pinky_dip",\
                       "x pinky_tip", "y pinky_tip", "z pinky_tip",\
                       "score de confiance"]
        
        header = [f"Frame n°", "cube blanc"] + cube_coords + ["cube rouge"] + cube_coords + ["premier cube foncé"] + cube_coords + ["second cube foncé"] + cube_coords + ["première main"] + hand_coords + ["seconde main"] + hand_coords + ["troisième main"] + hand_coords + ["quatrième main"] + hand_coords
        
        # Assemble the legend rows to be written in the csv file.
        legends_rows = [[f"Resultat pour la vidéo {os.path.basename(args.video_path).split('.')[0]}"],
                        [f"x1 : n° de colonne du point supérieur gauche de la boîte englobante"],
                        [f"y1 : n° de ligne du point supérieur gauche de la boîte englobante"],
                        [f"x2 : n° de colonne du point inférieur droite de la boîte englobante"],
                        [f"y2 : n° de ligne du point inférieur droite de la boîte englobante"],
                        [f"xc : n° de colonne du centre de la boîte englobante"],
                        [f"yc : n° de ligne du centre de la boîte englobante"]]
        
        # write in the csv file the legend rows and the header row.
        table_writer.writerows(legends_rows)
        table_writer.writerow(header)
        
    # Initialise the object containing the cubes detector with the model and the input shape specified in the arguments.
    cubes_detector = Cubes_Detector(args.model, input_shape)
    # Initialise the object containing the hands tracker with the maximum number of hands it has to tracks.
    hands_tracker = Hands_Tracker(num_hands)
    
    # Get the first frame of the video to work on.
    success, origin_img = vid.read()    
    
    # Initialise the progression bar
    with alive_bar(nb_frames-start-end, theme='classic') as bar :
            
        # Loop throught all the frames of the input video minus the ones that were asked to be ignored.
        for i in range(start, nb_frames-end) :
            
            # Print an error message if the video frame could not be read.
            if not success:
                print("Ignoring empty camera frame.")
                continue        
                
            # Ask the hands tracker to do detection and/or tracking of the hands on the current frame.
            hands_tracker.update(origin_img)
            
            if csv_write :
                # Create a one dimensional array representing one row of the csv file and which will receive all the features of the tracked objects. 
                table_row = np.full((297), None)
                # Initialise the variable which will receive the features of the tracked cubes.
            
            # Ask the cube detector to detect the cubes on the current video frame.
            cubes_detector.detect(origin_img)
            # If cubes were detected ask the detector to filter those detection instances in order to keep only the ones with the highest confidence score in each class.
            if cubes_detector.final_cls_inds.size > 0 :
                # If asked gather the cubes data to be written in the csv file.
                cubes_data = cubes_detector.filter_by_score(csv_write)
                                    
                # For each detected cube, draw on the frame the bounding box, label and confidence score of this cube.
                if display or write_video :
                    origin_img = cubes_detector.draw_on_image(origin_img, args.score_thr)
            
            # For each detected hands, draw on the frame the sqeleton and collect if asked the data to store in the csv file.
            hands_landmarks, hands_sides, hands_scores = hands_tracker.draw_and_export_csv(origin_img, csv_write, height, width)
                    
            # Assemble the row to be written in the csv file with all the data gathered.
            if csv_write :
                table_row[0:2] = [i, "||"]
                table_row[2:9] = cubes_data[0]
                table_row[9] = "||"
                table_row[10:17] = cubes_data[1]
                table_row[17] = "||"
                table_row[18:25] = cubes_data[2]
                table_row[25] = "||"
                table_row[26:33] = cubes_data[3]
                table_row[33] = "||"
                table_row[34] = hands_sides[0]
                table_row[35:98] = hands_landmarks[0]
                table_row[98] = hands_scores[0]
                table_row[99] = "||"
                table_row[100] = hands_sides[1]
                table_row[101:164] = hands_landmarks[1]
                table_row[164] = hands_scores[1]
                table_row[165] = "||"
                table_row[166] = hands_sides[2]
                table_row[167:230] = hands_landmarks[2]
                table_row[230] = hands_scores[2]
                table_row[231] = "||"
                table_row[232] = hands_sides[3]
                table_row[233:296] = hands_landmarks[3]
                table_row[296] = hands_scores[3]
                
                # Write in the csv file the row corresponding to the current frame.
                table_writer.writerow(table_row)

            # Write the processed frame in the output video file.
            if write_video :
                vid_out.write(origin_img)
                
            # If asked display the processed frame
            if display :
                cv.imshow("[COMBINED DETECTION]   Press \'q\' to quit  /  Press \'p\' to play/pause  /  Press \'f\' to skip 2 seconds  /  Press \'F\' to skip 5 seconds.", origin_img)
                
                key = cv.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    while True :
                        if cv.waitKey(1) == ord('p'):
                            break
                elif key == ord('f'):
                    for i in range(int(fps*2)):
                        success, origin_img = vid.read()
                elif key == ord('F'):
                    for i in range(int(fps*5)):
                        success, origin_img = vid.read()
                
            # Get the next frame.
            success, origin_img = vid.read()
            # Actualise the progress bar.
            bar()
    
    # Close the input video file.
    vid.release()
    # Close the output video file. 
    if write_video :
        vid_out.release()
    # Close the window that was displaying the processed frames.
    if display :
        cv.destroyWindow("[COMBINED DETECTION]   Press \'q\' to quit  /  Press \'p\' to play/pause  /  Press \'f\' to skip 2 seconds  /  Press \'F\' to skip 5 seconds.")