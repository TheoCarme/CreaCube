import sys
import numpy as np

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis



COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)



class Cubes_Detector():
    def __init__(self, model=r"C:\Users\tcarme\Documents\Stage_XLIM\YOLOX\yolox_s.onnx", input_shape=(640, 640)):
        super(Cubes_Detector, self).__init__()

        self.session = onnxruntime.InferenceSession(model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_shape = input_shape
        self.final_boxes = None
        self.final_scores = None
        self.final_cls_inds = None
    
    
    
    def detect(self, raw_img, conf=0.5) :
    
        final_boxes = final_scores = final_cls_inds = []
        img, ratio = preprocess(raw_img, self.input_shape)
            
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], self.input_shape, False)[0]
            
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1, class_agnostic=False)
        if dets is not None :
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            
        self.final_boxes = np.asarray(final_boxes)
        self.final_scores = np.asarray(final_scores)
        self.final_cls_inds = np.asarray(final_cls_inds)
    
    
    
    def filter_by_score(self, csv_write) :

        top_4_idx = np.array([], dtype=int)
        white_cube_id = red_cube_id = dark_cubes_idx = None
        cubes_data = None
        if csv_write :
            cubes_data = np.full((4,7), None)
        
        white_idx = np.asarray(np.where((self.final_cls_inds)==0.0))[0]
        red_idx = np.asarray(np.where((self.final_cls_inds)==1.0))[0]
        dark_idx = np.asarray(np.where(((self.final_cls_inds)==2.0) | ((self.final_cls_inds)==3.0)))[0]
        
        
        if white_idx.size > 0 :
            white_cube_id = np.argmax(self.final_scores[white_idx]) + np.min(white_idx)
            top_4_idx = np.concatenate((top_4_idx, white_cube_id), axis=None)
            
            
        if red_idx.size > 0 :
            red_cube_id = np.argmax(self.final_scores[red_idx]) + np.min(red_idx)
            top_4_idx = np.concatenate((top_4_idx, red_cube_id), axis=None)
                 
                 
        if dark_idx.size > 2 :
            dark_cubes_idx = np.argpartition(self.final_scores[dark_idx], -2)[-2:] + np.min(dark_idx)
        else :
            dark_cubes_idx = dark_idx
            
        if dark_idx.size > 0 :
            top_4_idx = np.concatenate((top_4_idx, dark_cubes_idx[0]), axis=None)
            
        if dark_cubes_idx.size > 1 :
            top_4_idx = np.concatenate((top_4_idx, dark_cubes_idx[1]), axis=None)
       
        self.final_boxes = self.final_boxes[top_4_idx]
        self.final_scores = self.final_scores[top_4_idx]
        self.final_cls_inds = self.final_cls_inds[top_4_idx]
        
        if csv_write :
        
            for box, score, cls in zip(self.final_boxes, self.final_scores, self.final_cls_inds) :
                
                cubes_data[int(cls)][0:4] = box
                cubes_data[int(cls)][4] = (box[0] + box[2])/2
                cubes_data[int(cls)][5] = (box[1] + box[3])/2
                cubes_data[int(cls)][6] = score
                
        return cubes_data
    
    
    
    def draw_on_image(self, image, conf) :
        
        if self.final_cls_inds is not None :
            image = vis(image, self.final_boxes, self.final_scores, self.final_cls_inds, conf=conf, class_names=COCO_CLASSES)
        
        return image
    
    
    
    def print_cubes(self) :
    
        if self.final_cls_inds is not None :
            for j, (box, score, cls) in enumerate(zip(self.final_boxes, self.final_scores, self.final_cls_inds)):
                print(f"Object n°{j} :\n\tbox = {box}\n\tscore = {score}\n\tclass = {cls}")
