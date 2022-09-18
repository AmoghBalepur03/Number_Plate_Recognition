#from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFilter
#METHOD 1

#*********************************************************************************************************************************************************************************


def conTO28x28(path):
  
  img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)   

  # convert each image of shape (32, 28, 1)
  w, h = img.shape
  if h > 28 or w > 28:
    (tH, tW) = img.shape
    dX = int(max(0, 28 - tW) / 2.0)
    dY = int(max(0, 28 - tH) / 2.0)

    img = cv2.copyMakeBorder(img, top=dY, bottom=dY,
          left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
          value=(0, 0, 0))
    img = cv2.resize(img, (28, 28))

  w, h = img.shape

  if w < 28:
      add_zeros = np.ones((28-w, h))*255
      img = np.concatenate((img, add_zeros))

  if h < 28:
      add_zeros = np.ones((28, 28-h))*255
      img = np.concatenate((img, add_zeros), axis=1)
  #img = np.expand_dims(img, axis=2)
  # Normalize each image
  cv2.resize(img , (32,32))
  return img

#METHOD 2
from PIL import Image
import os
import PIL
import glob

#*********************************************************************************************************************************************************************************


#for sklearn
def image_resize_sklearn(img):
    im1 = PIL.Image.open(img)

    image = im1.resize((32,32))
    #resizing image to 28,28
    image.save(img)
    #convert rgb to grayscale
    image = image.convert('L')
    image = np.array(image)
    #reshaping to support our model input 
    image = image.reshape(1,32,32,1)
    #normalizing
    image = image/255.0
    return image
    
#*********************************************************************************************************************************************************************************

    
def image_resize_model(img):
    im1 = PIL.Image.open(img)

    image = im1.resize((28,28))
    #resizing image to 28,28
    image.save(img)
    #convert rgb to grayscale
    image = image.convert('L')
    image = np.array(image)
    #reshaping to support our model input 
    image = image.reshape(1,28,28,1)
    #normalizing
    image = image/255.0
    return image

#*********************************************************************************************************************************************************************************

#to_plot_the_images
def plot_images(img1,  title1="",):
    fig = plt.figure(figsize=[18,18])
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)
    plt.show()

#*********************************************************************************************************************************************************************************









