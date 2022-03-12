import cv2
import sys
import numpy as np
import json
import os
from datetime import date

def letterbox_img(img, inp_dim):
    """Resize image with unchanged aspect ratio using padding.
    
    Args:
        img: (ndarray) Original image RBG with shape (H, W, C).
        inp_dim: (ndarray) with (width, height).
    
    Output:
        canvas: (ndarray) Array with desired size and contain the image at the center.
    """
    
    # get width and height of original image
    img_w, img_h = img.shape[1], img.shape[0]
    
    # get width and height of desierd dimension
    w, h = inp_dim 

    # calculate the new_w and new_h of content of image inside the desired dimension
    # calculate the scale_factor according to width or heigh --> get the minimum value
    # cause we want to maintain the information of orginal image.
    scale_factor = min(w / img_w, h / img_h)
    new_w = int(img_w * scale_factor)
    new_h = int(img_h * scale_factor)

    # resized orginal image
    # why cv2.INTER_CUBIC ?
    # https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
    # https://chadrick-kwag.net/cv2-resize-interpolation-methods/
    resized_img = cv2.resize(src = img, dsize = (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    
    # create the canvas
    canvas = np.full(shape = (inp_dim[1], inp_dim[0], 3), fill_value = 128)
    
    # paste the image on the canvas (at center)
    # canvas[top : bottom, left : right, :]
    top = (h - new_h) // 2
    bottom = top + new_h
    left = (w - new_w) // 2
    right = left + new_w
    canvas[top : bottom, left : right, :] = resized_img

    return canvas

def convert_bboxes_xyxy_to_xywh_COCO(bboxes_xyxy):
    """Convert bounding bxoes with xyxy to xywh format."""
    
    bboxes_xyxy = np.array(bboxes_xyxy)
    x = np.expand_dims(bboxes_xyxy[:, 0], 1)
    y = np.expand_dims(bboxes_xyxy[:, 1], 1)
    w = np.expand_dims(bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0], 1)
    h = np.expand_dims(bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1], 1)
    bboxes_xywh = np.hstack((x, y, w, h)).astype(np.float32).tolist()
    
    return bboxes_xywh

def _compute_area(bbox_xywh):
    return bbox_xywh[2] * bbox_xywh[3]

def _find_id(image_dir, image_file, json_file):
    image_file_id = None
    file_name = os.path.join(image_dir, image_file)
    for image_dct in json_file["images"]:
        if image_dct["file_name"] == file_name:
            image_file_id = image_dct["id"]
    
    return image_file_id

def rescale(bboxes, scale_factor):
    """
    Args:
        + bboxes: (list) with array shape (N, 4) with N is total of bboxes. 4 is represent to xywh.
        + scale_factor (tupe(int(), int())) with first index is scale_h, and second index is scale_w.
        *Note:* scale_h = height_new_image_shape / height_original_shape.
                scale_w = width_new_image_shape / width_original_shape
    """
    
    bboxes = np.asarray(bboxes)
    scale_array = np.array([scale_factor[1], scale_factor[0], scale_factor[1], scale_factor[0]])
    bboxes /= scale_array
    return bboxes.tolist()

def _save_not_exist(image_dir, image_files, save_file, bboxes, class_ids, class_labels, scale_factor):
    # resacle bboxes
    bboxes = rescale(bboxes, scale_factor)

    # dict()
    info_dict = {"description" : "Interview",
                 "url" : "https://github.com/DatacollectorVN",
                 "version" : 1,
                 "year" : 2022,
                 "contributor" : "DatacollectorVN"
                }
    
    # list(dict())
    lincenses = [{"url" : "https://github.com/DatacollectorVN",
                  "id" : 1,
                  "name" : "DatacollectorVN"
                 }
                ]

    # list(dict())
    categories = []
    for class_label in class_labels:        
        category_dict =  {"supercategory" : "car color",
                          "id" : class_labels.index(class_label),
                          "name" : class_label,
                         }
        categories.append(category_dict)

    json_file = {"info" : info_dict,
                     "licenses" : lincenses,
                     "categories" : categories,
                    }
    image_lst = []
    annotations_lst = []
    bbox_id_cum = 0
    for  i, image_file in enumerate(image_files):
        h, w, _ = cv2.imread(os.path.join(image_dir, image_file)).shape
        id_ = i + 1
        image_dct = {"id" : id_,
                     "license" : 1,
                     "file_name" : os.path.join(image_dir, image_file),
                     "coco_url" : "",
                     "height" : h,
                     "width" : w,
                     "date_captured" : date.today().strftime("%d/%m/%Y"),
                     "flickr_url" : "",
                    }
        image_lst.append(image_dct)

        bboxes_per_image = bboxes[i]
        class_ids_per_image = class_ids[i]
        for j, bbox_per_image in enumerate(bboxes_per_image):
            bbox_id = bbox_id_cum + j + 1
            class_ = class_ids_per_image[j]
            area = _compute_area(bbox_xywh = bbox_per_image)
            annotations = {"id" : bbox_id,
                            "image_id" : id_,
                            "iscrowd" : 0, 
                            "category_id" : class_labels[class_],
                            "segmentation" : [],
                            "bbox" : bbox_per_image,
                            "area" : area
                            }
            annotations_lst.append(annotations)
            # reset value bbox_id_cum
            if j == len(bbox_per_image) - 1:
                bbox_id_cum = bbox_id_cum + j + 1
    json_file["images"] = image_lst
    json_file["annotations"] = annotations_lst
    
    with open(save_file, 'w') as file:
        json.dump(json_file, file)

def _save_exist(image_dir, image_files, save_file, bboxes, class_ids, class_labels, json_file, scale_factor):
    # resacle bboxes
    bboxes = rescale(bboxes, scale_factor)

    for i, image_file in enumerate(image_files):
        bbox_id_start = 0
        image_file_id = _find_id(image_dir, image_file, json_file)
        
        # if image_file_id exist, then update new object in that image.
        if image_file_id: 
            id_ = image_file_id
            annotation_exist_lst = []
            for annotation in json_file["annotations"]:
                if annotation["image_id"] == image_file_id:
                    annotation_exist_lst.append(annotation["id"])
            
            bbox_id_start = max(annotation_exist_lst) + 1
        else:
            image_id_lst = []
            for annotation in json_file["annotations"]:
                image_id_lst.append(annotation["image_id"])
            
            id_ = max(image_id_lst) + 1
            h, w, _ = cv2.imread(os.path.join(image_dir, image_file)).shape
            image_dct = {"id" : id_,
                         "license" : 1,
                         "file_name" : os.path.join(image_dir, image_file),
                         "coco_url" : "",
                         "height" : h,
                         "width" : w,
                         "date_captured" : date.today().strftime("%d/%m/%Y"),
                         "flickr_url" : "",
                        }
            # update new image information in images field.
            json_file["images"].append(image_dct)
        
        # Update new object in images correspoding to image_id  
        bboxes_per_image = bboxes[i]
        class_ids_per_image = class_ids[i]
        for j, bbox_per_image in enumerate(bboxes_per_image):
            bbox_id = bbox_id_start + j + 1
            class_ = class_ids_per_image[j]
            area = _compute_area(bbox_xywh = bbox_per_image)
            annotations = {"id" : bbox_id,
                            "image_id" : id_,
                            "iscrowd" : 0, 
                            "category_id" : class_labels[class_],
                            "segmentation" : [],
                            "bbox" : bbox_per_image,
                            "area" : area
                          }
            json_file["annotations"].append(annotations) 

        with open(save_file, 'w+') as file:
            json.dump(json_file, file)

def save_dataset_COCO(image_dir, image_files, save_file, bboxes, class_ids, class_labels, scale_factor):
    """Save dataset with COCO format
    Args:
        + image: (ndarray) Image array with shape (H, W, C) 
        + image_dir: (str) Path of images directory.
        + image_files: (list) List of image file ids.
        + save_file: (str) Name of save file.
        + bboxes: (list) Offset-values of bounding boxes. array shape (n, m, 4)
        + class_ids: (list) Class ids correspoding to offset-values of bounding boxes. (n, m)
        + class_labels: (list) Class labels default of dataset.
        + scale_factor(tuple(int(), int())) scale_factor of image when labeling.
        *Note:* n number of images and m number of objects per image.
    """
    
    if os.path.isfile(save_file):
        f = open(save_file)
        json_file = json.load(f)
        _save_exist(image_dir, image_files, save_file, bboxes, class_ids, class_labels, json_file, scale_factor)
    else:
        _save_not_exist(image_dir, image_files, save_file, bboxes, class_ids, class_labels, scale_factor)
