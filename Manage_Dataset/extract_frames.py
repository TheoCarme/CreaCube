import cv2 as cv
import argparse
import os
from alive_progress import alive_bar
from yolox.utils import mkdir


def make_parser():
    parser = argparse.ArgumentParser("extract frames")
    parser.add_argument(
        "-i",
        "--input_video",
        type=str,
        help="Path to your input video file from which frames will be extracted.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="Extracted_Frames",
        help="Path to your output directory in which extracted frames will be written.",
    )
    parser.add_argument(
        "-n",
        "--nb_frames",
        type=int,
        default=100,
        help="The number of frames you want to extract from the video.",
    )
    
    return parser


if __name__ == '__main__':

    args = make_parser().parse_args()
    
    input_video = args.input_video
    output_dir = args.output_dir
    nb_frames = args.nb_frames
    
    cap = cv.VideoCapture(input_video)
    mkdir(output_dir)
            
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    interval = length // nb_frames
    digit_nb = len(str(length))
    count = 1
    pictures_counter = 0

    # Initialise the progression bar
    with alive_bar(nb_frames, theme='classic') as bar :
        while cap.isOpened() and pictures_counter < nb_frames :
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
                
            if count % interval == 0 :
                filename = output_dir + '\\' + os.path.basename(args.input_video).split('.')[0] + f"#{count:0{digit_nb}d}.png"
                cv.imwrite(filename, frame)
                pictures_counter += 1
                # Actualise the progress bar.
                bar()
            
            count += 1
                                
        cap.release()
