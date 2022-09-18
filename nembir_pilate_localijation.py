import cv2 as cv
import numpy as np
from numpy.lib.function_base import delete
from imaje_resije import *

'''****************************************************************************************'''
# IMAGE_ENHANCING_FUNCTIONS
def D_filter(image):
    return cv.filter2D(image , -1 , np.ones((1,1), np.float32)/1)


def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

def get_greyscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def thresholding2(image):
    return cv.threshold(image, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

def canny(image):
    return cv.Canny(image, 100, 200)
    
def remove_noise(image):
    return cv.medianBlur(image, 3)

def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

def noiseRemoval(image):
    return cv.bilateralFilter(image, 11, 17, 17)

#def histogramEqualization(image):
    #return cv.absdiffequalizeHist(image)

def morphologicalOpening(image, structElem):
    return cv.morphologyEx(image, cv.MORPH_OPEN, structElem, iterations=15)

def subtractOpenFromHistEq(histEqImage, morphImage):
    return cv.subtract(histEqImage, morphImage)

def tresholding(image):
    x,t=cv.threshold(image, 127, 255, cv.THRESH_BINARY, cv.THRESH_OTSU)
    return t

def remove_shadow(image):
    rgb_planes = cv2.split(image)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        diff_img = 255 - cv2.absdiff(plane, dilated_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        return result_norm


#***************************************************************************************************************************************************************#
# Edge Detection
def edgeDetection(image, threshold1 = 2, threshold2= 10):
    cannyImage = cv.Canny(image, threshold1, threshold2)
    cannyImage = cv.convertScaleAbs(cannyImage)
    return cannyImage


def imageDilation(image, structElem):
    return cv.dilate(image, structElem, iterations=1)

#Contours
def findContours(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ''' For this problem, number plate should have contours with a small area as compared to other contours.
        Hence, we sort the contours on the basis of contour area and take the least 10 contours'''
    return sorted(contours, key=cv.contourArea, reverse=True)[:10]

#Ramer-Douglas-Peucker Algorithm
def approximateContours(contours):
    approximatedPolygon = None
    for contour in contours:
        contourPerimeter = cv.arcLength(contour, True)
        approximatedPolygon = cv.approxPolyDP(contour, 0.06*contourPerimeter, closed=True)
        if(len(approximatedPolygon) == 4):                              #number plate is usually a rectangle
            break                                                       #therefore breaks when a quad is encountered
    return approximatedPolygon


# Highlighting the image
def drawLocalizedPlate(image, approximatedPolygon):
    M=cv.moments(approximatedPolygon)
    cX=int(M["m10"]/M["m00"])
    cY=int(M["m01"]/M["m00"])
    
    finalImage = cv.drawContours(image, [approximatedPolygon], -1, (0, 255, 0), 3)
    
    cv.circle(finalImage, (cX, cY), 7, (0, 255, 0), -1)
    cv.putText(finalImage, "Centroid of Plate: ("+str(cX)+", "+str(cY)+")", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return finalImage
#clears the plates folder 
def delete_folder():
    import os, shutil
    folder = 'plates'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
             print('Failed to delete %s. Reason: %s' % (file_path, e))

#**************************************************************************************************************************************************************#
#MAIN_FUNCTION
def main(path):
    #clearing the plates folder everytime the function is used
    delete_folder()

    #using functions to enhance image quality
    

    image=cv.imread(path)
    image=D_filter(image)
    image=grayscale(image)
    image=tresholding(image)
    cnts = findContours(image)
    
    #plot_images(image)

    plates=[] # incase multiple possible plates are recognized 
    plate = None
    #finding contours
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        edges_count = cv2.approxPolyDP(c, 0.02* perimeter, True)
        if len(edges_count) == 4:
            #finding a quadrilateral
            x,y,w,h = cv2.boundingRect(c)
            plate = image[y:y+h, x:x+w]
            plates+=[plate]
            #there may be more than one possible "numberplates "
    for i in range(len(plates)):
        cv.imwrite( f'plates/plates{i}.png', plates[i])
        

#**************************************************************************************************************************************************************#


    


