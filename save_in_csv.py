from PIL import Image
import numpy as np
import os, os.path, time
import matplotlib.image as img
import csv
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
import cv2
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
from nembir_pilate_localijation import main as number_plate_localizer
from save_in_csv import *

model = load_model(r"Combined_Resnet_50_Epochs")
     

#********************************************************************************************************************************************************************************* 

def predict_image(img):
  image = cv2.imread(img)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  edged = cv2.Canny(blurred, 30, 150)
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sort_contours(cnts, method="left-to-right")[0]
  chars = []
  for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    # filter out bounding boxes, ensuring they are neither too small
    # nor too large
    if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
      # extract the character and threshold it to make the character
      # appear as *white* (foreground) on a *black* background, then
      # grab the width and height of the thresholded image
      roi = gray[y:y + h, x:x + w]
      thresh = cv2.threshold(roi, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      (tH, tW) = thresh.shape
      # if the width is greater than the height, resize along the
      # width dimension
      if tW > tH:
        thresh = imutils.resize(thresh, width=32)
      # otherwise, resize along the height
      else:
        thresh = imutils.resize(thresh, height=32)
      (tH, tW) = thresh.shape
      dX = int(max(0, 32 - tW) / 2.0)
      dY = int(max(0, 32 - tH) / 2.0)
      # pad the image and force 32x32 dimensions
      padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0))
      padded = cv2.resize(padded, (32, 32))
      padded = padded.astype("float32")
      padded = np.expand_dims(padded, axis=-1)
      chars.append((padded, (x, y, w, h)))
      cv2.imshow("ok" ,padded)
	 #final_image=np.expand_dims(padded,axis=2)
  #print(chars)
  boxes = [b[1] for b in chars]
  #cv2_imshow(chars)
  chars = np.array([c[0] for c in chars], dtype="float32")
  # OCR the characters using our handwriting recognition model
  preds = model.predict(chars)
  # define the list of label names
  labelNames = "0123456789"
  labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  labelNames = [l for l in labelNames]
  output=""
  for (pred, (x, y, w, h)) in zip(preds, boxes):
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]
    output+=label
  
  return output

#*********************************************************************************************************************************************************************************

def save_in_csv():
    csv_path=r'images\images.csv'
    format='.png'
    myDir = "plates"
    def createFileList(myDir, format='.png'):
        fileList = []
        for root, dirs, files in os.walk(myDir, topdown=False):
                for name in files:
      #             print(name)
                   if name.endswith(format):
                      fullName = os.path.join(root, name)
                      fileList.append(fullName)
        return fileList



    for image in createFileList(myDir):
        import numpy as np
        import matplotlib.image as img
        imageMat = img.imread(image)
       # print("Image shape:", imageMat.shape)
        if len(imageMat.shape) == 2:
            x,y=imageMat.shape
            image_mat = imageMat.reshape(x,y,-1)
        else:
            image_mat=imageMat
 
# if image is colored (RGB)
        if(image_mat.shape[2] != -1):  
  # reshape it from 3D matrice to 2D matrice
            imageMat_reshape = image_mat.reshape(image_mat.shape[0],
                                      -1)
           # print("Reshaping to 2D array:",
           # imageMat_reshape.shape)
# if image is grayscale
        else:
            imageMat_reshape = image_mat
     

        with open(csv_path , "a") as csv_file:
           writer=csv.writer(csv_file)
           writer.writerow([predict_image(image) ,'\t' , imageMat_reshape , '\n' ])
    
#*********************************************************************************************************************************************************************************
         

