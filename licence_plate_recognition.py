import cv2
import numpy as np
import glob
import random
import imutils
import pytesseract
from PIL import Image


# Load Yolo
net = cv2.dnn.readNet("yolov4_best.weights", "yolov4.cfg")

# Name custom object
classes = ["licence plate"]


layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Loading image, replace below address in cv2.imread() with your image_path
img = cv2.imread('/home/imran_afreed/Desktop/winter_intern/ocr/licence_plate_recognition/images/two.jpeg')
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.85:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        tmp = img[y:y+h,x:x+w]
        # cv2.rectangle(img, (x, y), (x + w, y + h), (20, 255, 10), 1)
        tmp = cv2.resize(tmp, None, fx=2.5, fy=2.3)
        cv2.imwrite('./cropped.png',tmp)


#loading cropped license plate
crp = cv2.imread("cropped.png") 

#converting cropped image to gray scale
gray = cv2.cvtColor(crp,cv2.COLOR_BGR2GRAY) 

#generating binary image from gray scale image useing adaptive thereshold method
adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 0)

#generating contours to detect the unwanted border
#uncomment the below comment to see contours
count,_ = cv2.findContours(adaptive_threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
count_img = cv2.drawContours(adaptive_threshold,count,-1,(255,0,0),2)
# cv2.imshow(" all counts", count_img)

#index of countour with most area
c  = max(count, key = cv2.contourArea)

#cooridnated of boudning box with maximum area
x,y,w,h = cv2.boundingRect(c)
# count_img = cv2.rectangle(adaptive_threshold,(x,y),(x+w,y+h),(0,255,0),2)

#cropping
count_img = count_img[y+2:y+h-1,x+9:x+w-5]

#increasing font thickness
kernel = np.ones((1,1),np.uint8)
count_img = cv2.erode(count_img,kernel,iterations = 1)

#saving final processed image
cv2.imwrite("edges cropped.png",count_img)

#scaling image before performing OCR
count_img = cv2.resize(count_img,None, fx = 0.4, fy = 0.3)


#image is convereted into straing
config = "--psm 3"
text = pytesseract.image_to_string(count_img, config=config, lang="eng")
print("lince number of current vehicle is " + text)


key = cv2.waitKey(0)  
cv2.destroyAllWindows()