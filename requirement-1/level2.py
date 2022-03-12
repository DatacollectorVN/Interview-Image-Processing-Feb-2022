import cv2
import numpy as np
import sys
import argparse
import yaml
import os
from src.utils import letterbox_img, convert_bboxes_xyxy_to_xywh_COCO, save_dataset_COCO
from pathlib import Path

FILE_CONFIG = os.path.join("config", "dataset.yaml")
with open(FILE_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def draw_rectangle(action, x, y, flags, *userdata):
    
    # Referencing global variables 
    global xmin_ymin, xmax_ymax, offset_values
    
    # Mark the top left corner, when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        xmin_ymin = [x,y]

    # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        xmax_ymax = [x,y]
        offset_values_opencv.append([xmin_ymin, xmax_ymax])
        cv2.rectangle(image, xmin_ymin, xmax_ymax, (0, 255, 0), 2, 8)
        cv2.imshow(window_name, image)
    
def main(image_file):
    global image, window_name, offset_values_opencv
    image_dir = os.path.join(image_file.split('/')[0], image_file.split('/')[1])
    label = input("Enter class label: ")
    offset_values_opencv = []
    image = cv2.imread(image_file)
    scale_factor = (1, 1) # (scale_h, scale_w)
    if image.shape[0] > 1000 or image.shape[1] > 1000:
        inp_dim = (1000, 1000)
        scale_h = inp_dim[0] / image.shape[0]
        scale_w = inp_dim[1] / image.shape[1]
        scale_factor = (scale_h, scale_w)
        image = letterbox_img(image, inp_dim = inp_dim).astype(np.uint8)
    window_name = Path(image_file).stem
    image_ori = image.copy()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle)
    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(0)
        
        # if the F5 key is pressed, reset the cropping region
        if key == 194: 
            offset_values_opencv = [] 
            image = image_ori.copy()
        
        # if the esc key is pressed, break from the loop
        elif key == 27:
            break
    
    labels = [label for _ in range(len(offset_values_opencv))]
    bboxes_xyxy = []
    for coordinate in offset_values_opencv:
        bboxes_xyxy.append([coordinate[0][0], coordinate[0][1], coordinate[1][0], coordinate[1][1]])
    
    bboxes_xywh = convert_bboxes_xyxy_to_xywh_COCO(bboxes_xyxy)
    bboxes_xywh = [bboxes_xywh]
    class_ids = [params["CLASS_LABELS"].index(label) for label in labels]
    class_ids = [class_ids]
    image_dir = os.path.join(image_file.split('/')[0], image_file.split('/')[1])
    save_dataset_COCO(image_dir, [Path(image_file).stem + ".jpg"], "test.json", 
                      bboxes_xywh, class_ids, params["CLASS_LABELS"], scale_factor)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--imagefile', dest = 'imagefile', type = str,
                        default = None, help = "Image file path") 
    
    args = parser.parse_args()
    image_file = args.imagefile
    main(args.imagefile)