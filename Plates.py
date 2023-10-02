from PIL import Image
from cv2 import imread
from nembir_pilate_localijation import *
import cv2
import pytesseract as pt
pt.pytesseract.tesseract_cmd = r"C:\Users\AB1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

image="Cars\car_2.jpg"
img = cv2.imread(image)
image=D_filter(img)
image = tresholding(image)
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
    cv2.imwrite( f'plates_2/plates{i}.png', plates[i])


img = Image.open(image)
text_ = pt.image_to_string(img)
print(text_)