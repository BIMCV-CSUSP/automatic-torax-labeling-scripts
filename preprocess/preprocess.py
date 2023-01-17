import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch

from skimage import exposure
import skimage.morphology
from skimage.morphology import disk

import math
from shutil import copyfile

import multiprocessing
from preprocess.yolov5 import detect as yolo_detect
# Load a png file convertd from DICOM
def load_file(filename, monochrome):  
    # Save image in set directory
    # Read RGB image
    pixels = cv2.imread(filename, cv2.IMREAD_UNCHANGED) 
    if pixels is None:
        return None, None

    if monochrome == "MONOCHROME1":
        pixels = np.amax(pixels) - pixels

    pixels_eq = exposure.equalize_adapthist(pixels)
    pixels_eq = (pixels_eq * 255).astype(np.uint8)
    
    pixels = pixels - np.min(pixels)
    pixels = pixels / np.max(pixels)
    pixels = (pixels * 255).astype(np.uint8)
    
    return pixels, pixels_eq

# Convert YOLO coords to pixel coords
def box2coords(x,y,w,h,image_w, image_h):
    x1, y1 = round((x-w/2) * image_w), round((y-h/2) * image_h)
    x2, y2 = round((x+w/2) * image_w), round((y+h/2) * image_h)
    return x1, y1, x2, y2

# Function to get the BB data from the images DF
def get_boxes(image_id):
    
    image = image_id.replace('.png','_image')
    ti = images_df[images_df['id'] == image]
    bx = [[],[]]
    bx[0] = [0,0,0,0,""]
    bx[1] = [0,0,0,0,""]
    
    if str(ti['boxes'].values[0]) != "nan":
        box = str(ti['boxes'].values[0]).replace("'","\"")
        boxes = json.loads(box)
        lab = ti['label'].values[0].split(" ")
        i = 0
        for b in boxes:
            bx[i] = [int(b['x']), int(b['y']), int(b['width']),int(b['height']),lab[0]]
            i = i+1
    return bx

# This function draws boxes on images, one line at a time
def draw_boxes(boxes, z):

    for i in boxes:     
        # Top
        x = [i[0] - z[0], i[0] + i[2] - z[0]]        # [ x1 , x2 ]
        y = [i[1] - z[1], i[1] - z[1]]               # [ y1 , y2 ]
        plt.plot(x,y, color='#ff8838', linewidth=2)
        
        # Bottom
        y = [i[1] + i[3] - z[1], i[1] + i[3] - z[1]]
        plt.plot(x,y, color='#ff8838', linewidth=2)
        
        # Left
        x = [i[0] - z[0], i[0] - z[0]]
        y = [i[1] - z[1], i[1] + i[3] - z[1]]
        plt.plot(x,y, color='#ff8838', linewidth=2)

        # Right         
        x = [i[0] + i[2] - z[0], i[0] + i[2] - z[0]]
        plt.plot(x,y, color='#ff8838', linewidth=2)
        
# Call YOLO detect.py with the image we exported
def detect(img_path, run_path):
    # Clean up results from the last run
    name = img_path.split('/')[-1].split('.')[0]
    path = os.path.join(run_path,'exp','labels',f'{name}.txt')
    if os.path.exists(path):
        os.remove(path)

    # Call yolo detect
    #os.system(f"source venv-BiomedicalRX-Torax/bin/activate && python yolov5/detect.py --source {img_path} --project {run_path} --weights anatomy_detection.pt --img 640 --exist-ok --line-thickness 10 --save-txt > /dev/null 2>&1 && deactivate ")
    #!source venv-BiomedicalRX-Torax/bin/activate && python yolov5/detect.py --source test.jpg --weights anatomy_detection.pt --img 640 --exist-ok --line-thickness 10 --save-txt && deactivate
    yolo_detect.run(source=img_path, project=run_path, weights = "anatomy_detection.pt", imgsz=640, exist_ok=True, line_thickness=10, save_txt=True)
    
def crop(image, run_path):

    # Export a JPG for YOLO
    img_path = os.path.join(run_path,'img.jpg')
    cv2.imwrite(img_path, image);

    # Run detect
    detect(img_path, run_path)

    # Get the predicted boxes into a dataframe
    datapath = os.path.join(run_path,'exp','labels', 'img.txt')
    if not os.path.exists(datapath):
        return None, None
    boxes = pd.read_csv(datapath, delim_whitespace=True, header=None, index_col=False)

    # Convert the normalized YOLO BB data of the predicted anatomy to pixel coords and add them to the dataframe
    for index, row in boxes.iterrows():

        x, y, xx, yy = box2coords(row[1], row[2], row[3], row[4], image.shape[1], image.shape[0])
        boxes.at[index,'x'] = x
        boxes.at[index,'y'] = y
        boxes.at[index,'xx'] = xx
        boxes.at[index,'yy'] = yy

    #check if we have both lungs
    if len(boxes[(boxes[0] == 0)]) == 0 or len(boxes[(boxes[0] == 1)]) == 0:
        return None, None

    # Get only the lung coords .. class 0 and 1 .. you could choose shoulders or clavicles instead .. or all of them
    lungs = boxes[(boxes[0] == 0) | (boxes[0] == 1)]
    lungs.head()

    # Figure out the max dimensions of the predicted anatomy, and crop the image to those coords
    x1 = int(lungs['x'].min())
    x2 = int(lungs['xx'].max())
    y1 = int(lungs['y'].min())
    y2 = int(lungs['yy'].max())

    #cropped = image[y1:y2, x1:x2]

    # Get the annotated BB coordinates
    #bb = get_boxes(str(os.path.basename(filename)))

    return (x1,y1),(x2,y2)


def equalize(image):    
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(image)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) # MORPH_ELLIPSE
    eq_img = cv2.equalizeHist(image)
    tophat = cv2.morphologyEx(eq_img, cv2.MORPH_TOPHAT, kernel)
    bothat = cv2.morphologyEx(eq_img, cv2.MORPH_BLACKHAT, kernel)
    hat_img = eq_img + tophat - bothat
    
    return np.dstack((image, clahe_img, hat_img))



def preprocess(filename, monochrome, temp_path = '/tmp/yolo'):
    temp_subpath = temp_path
    if not os.path.exists(temp_path):
        os.makedirs(temp_subpath)
    
    pixels, pixels_eq = load_file(filename, monochrome)
    if pixels is None:
        return None

    p1, p2 = crop(pixels_eq, temp_subpath)

    if p1 == None:
        return None
    
    cropped = pixels[p1[1]:p2[1], p1[0]:p2[0]]
    preprocessed = equalize(cropped)

    return preprocessed

