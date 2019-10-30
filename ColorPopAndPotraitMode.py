"""
Mask R-CNN is the model used for instane segmentation for the project

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by M S Sreenidhi Iyengar

------------------------------------------------------------

Before running the script see and install the pre-reqsites 

Usage : Run the program command line as such:

    # ColorPop for an image
    python ColorPopAndPotraitMode ColorPop --image {PATH TO IMAGE} 

    # ColorPop for an video
    python ColorPopAndPotraitMode ColorPop --video {PATH TO VIDEO} 

    # PotraitMode for an image
    python ColorPopAndPotraitMode PotraitMode --image {PATH TO IMAGE}

    # PotraitMode for a video
    python ColorPopAndPotraitMode PotraitMode --video {PATH TO VIDEO}

    # PotraitMode for an image with user specfied Gausian Blur Kernel
    python ColorPopAndPotraitMode PotraitMode --image {PATH TO IMAGE} --blurkernel 69,69

    # PotraitMode for a video with user specfied Gausian Blur Kernel
    python ColorPopAndPotraitMode PotraitMode --image {PATH TO VIDEO} --blurkernel 42,42

    
"""

# Importing the required libraries
import tensorflow as tf
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import datetime

# To hide the deprication and other tf warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# print(ROOT_DIR) This must be Mark_RCNN directory for all imports to work correctly
# Import Mask RCNN
# To find local version of the library
sys.path.append(ROOT_DIR) 
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
# To find local version
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Apply_mask_potrait_mode function that applies the blur to the original image in predicted areas
# To the function we pass original image and potrait image
# Mix them both according to the masks predicted the Mask_RCNN model and return image
# Since color image contains 3 channels we apply blur to all 3 color channels 
def apply_mask_potrait_mode(image, potrait_mode, mask):
    image[:, :, 0] = np.where(
        mask == 0,
        potrait_mode[:, :,0],
        image[:, :, 0])
    image[:, :, 1] = np.where(
        mask == 0,
        potrait_mode[:, :,1],
        image[:, :, 1])
    image[:, :, 2] = np.where(
        mask == 0,
        potrait_mode[:, :,2],
        image[:, :, 2])
    return image

# Apply_mask_color_pop function that applies the grayscale to the original image in predicted areas
# To the function we pass original image and grayscale image
# Mix them both according to the masks predicted the Mask_RCNN model and return image
# Since color image contains 3 channels we apply gary to all 3 contains whereas gray image is single channel image
def apply_mask_color_pop(image, color_pop, mask):
    image[:, :, 0] = np.where(
        mask == 0,
        color_pop[:, :],
        image[:, :, 0])
    image[:, :, 1] = np.where(
        mask == 0,
        color_pop[:, :],
        image[:, :, 1])
    image[:, :, 2] = np.where(
        mask == 0,
        color_pop[:, :],
        image[:, :, 2])
    return image

# Detect_regions_and_apply_masks function used for calcuting object detection results from thr original image
# Function finds the largest mask for the person and sends it to apply_mask_color_pop or apply_mask_potrait_mode function 
def detect_regions_and_apply_masks(image, boxes, masks, ids, names, scores, color_pop, potrait_mode):
    max_area = 0
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        square = (y2 - y1) * (x2 - x1)

        label = names[ids[i]]
        if label == 'person':
            if square > max_area:
                max_area = square
                mask = masks[:, :, i]
            else:
                continue
        else:
            continue
    if args.command.lower() == "colorpop": 
        image = apply_mask_color_pop(image, color_pop, mask)
    elif args.command.lower() == "potraitmode":
        image = apply_mask_potrait_mode(image, potrait_mode, mask)
    return image

# Resize function is used to decrease the size of the image
# It preserves the aspect ratio of the image by using scale_percent
# Any output image dimensions will be equal or under dim = (2560,1440)

def resize(image, scale_percent):
    scale_percent =  scale_percent
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

# Image_util function will try to limit the scale_precent of the image 
# It is useful to obtain largest image under dim = (2560,1440)
def image_util(image):
    scale_percent = 90
    while True:
        image = resize(image, scale_percent)
        # To check the resized shape and scale percent of image 
        # print(image.shape, scale_percent) 
        if image.shape[0] <= 2560 and image.shape[1] <= 1440:
            return image, scale_percent
        scale_percent -= 10

# Color_Pop
def ColorPop(image_path=None, video_path=None):

    if image_path:
        image_path_dir = image_path.split('.')[0] 
        image_path = image_path
        img = cv2.imread(image_path)
        print('Dimensions         : ',img.shape)
        image, sp = image_util(img)
        print("Resized Dimensions : ",image.shape)
        print(f'Scale Percentage   : {sp}%')

        color_pop = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Gray resolution    : ",color_pop.shape)

        results = model.detect([image], verbose=0)
        r = results[0]
        frame = detect_regions_and_apply_masks(
                        image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], color_pop, color_pop
                )
        file_path = image_path_dir+"ColorPop_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        cv2.imwrite(file_path, frame)
        print('Created Color Pop image check in the Directory : ',file_path)

    elif video_path:
        video_path_dir = video_path.split('.')[0]
        video_path = video_path
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = video_path_dir+"ColorPop_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                color_pop = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                video_frame = detect_regions_and_apply_masks(
                                image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], color_pop, color_pop
                        )
                # RGB -> BGR to save image to video
                video_frame = video_frame[..., ::-1]
                # Add image to video writer
                vwriter.write(video_frame)
                count += 1
        vwriter.release()
        print("Created Color Pop video check in the Directory: ", file_path)

    

def PotraitMode(image_path=None, video_path=None, blur_kernel=None):
    print(type(blur_kernel), blur_kernel)
    if blur_kernel==None:
        blur_kernel = tuple([99,99])
    else:
        h = int(blur_kernel.split(',')[0])
        w = int(blur_kernel.split(',')[1])
        blur_kernel = tuple([h,w])
    print(type(blur_kernel), blur_kernel)

    if image_path:
        image_path_dir = image_path.split('.')[0] 
        image_path = image_path
        img = cv2.imread(image_path)
        print('Dimensions         : ',img.shape)
        #dim = (2560,1440)
        image, sp = image_util(img)
        print("Resized Dimensions : ",image.shape)
        print(f'Scale Percentage   :  {sp}%')
        potrait_mode = cv2.GaussianBlur(image,blur_kernel,cv2.BORDER_DEFAULT)
        print("Potrait resolution : ", potrait_mode.shape)
        print("Blur Kernel Size   : ", blur_kernel)

        results = model.detect([image], verbose=0)
        r = results[0]
        frame = detect_regions_and_apply_masks(
            image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],potrait_mode, potrait_mode
        )
        file_path = image_path_dir+"PotraitMode_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        cv2.imwrite(file_path, frame)
        print('Created Potrait Mode image check in the Directory: ', file_path)

    elif video_path:
        video_path_dir = video_path.split('.')[0]
        video_path = video_path
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = video_path_dir+"PotraitMode_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                potrait_mode = cv2.GaussianBlur(image,blur_kernel,cv2.BORDER_DEFAULT)
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                video_frame = detect_regions_and_apply_masks(
                                image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], potrait_mode, potrait_mode
                        )
                # RGB -> BGR to save image to video
                video_frame = video_frame[..., ::-1]
                # Add image to video writer
                vwriter.write(video_frame)
                count += 1
        vwriter.release()
        print("Created Potrait Mode video check in the Directory: ", file_path)

'''
func = int(input('1.ColorPop 2.potrait_modeMode Enter your choice : '))
if func == 1:
    ColorPop()
'''
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Image Segemntation Mask R-CNN Model to create Colorpop or PotraitMode.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'ColorPop' or 'PotraitMode'")
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply Color Pop or Potrait Mode on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply Color Pop or Potrait Mode on')
    parser.add_argument('--blurkernel', required=False,
                        metavar="Two integer values 69,69",
                        help='To create different intensity of Potrait Mode effect')
    args = parser.parse_args()

    # Validate arguments
    #if args.command == "Colorpop" or 'potrait_modeMode':
    #    assert args.image or args.video, "Provide --image or --video to apply ColorPop or potrait_modeMode"

    # Configurations
    if args.command.lower() == "colorpop" or 'potraitmode':
        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1 

        config = InferenceConfig()
    
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=MODEL_DIR)

    weights_path = COCO_MODEL_PATH
    # Load weights
    #print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    #Apply Effect
    if args.command.lower() == "colorpop":
        ColorPop(image_path=args.image, video_path=args.video)
    elif args.command.lower() == "potraitmode":
        PotraitMode(image_path=args.image,video_path=args.video, blur_kernel = args.blurkernel)
    else:
        print("'{}' is not recognized. "
              "Use 'ColorPop' or 'PotraitMode'".format(args.command))
