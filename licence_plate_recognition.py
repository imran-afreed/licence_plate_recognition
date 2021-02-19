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
img = cv2.imread('/home/imran_afreed/Desktop/licence_plate_recognition/images/one.jpg')
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


crp = cv2.imread("cropped.png")
crp = cv2.resize(crp, None, fx=0.5, fy=0.35)

#Gray scalling cropped.png
gray = cv2.cvtColor(crp,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray scale image before cropping", gray)

#converting gray scale image to binary
adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 0)

#drawing contours
count,_ = cv2.findContours(adaptive_threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
count_img = cv2.drawContours(adaptive_threshold,count,-1,255,2)

# count_img = adaptive_threshold
c  = max(count, key = cv2.contourArea)

#coordinates and dimensions of rectanagular bounding box
x,y,w,h = cv2.boundingRect(c)

#cropping boundries
gray = gray[y+1:y+h-1,x+1:x+w-1]


config = "--psm 3"
text = pytesseract.image_to_string(gray, config=config, lang="eng")
print("licence plate number detected is " + text)

cv2.imwrite("gray.png", gray)
cv2.imshow("final image used for OCR", gray)

key = cv2.waitKey(0)  
cv2.destroyAllWindows()