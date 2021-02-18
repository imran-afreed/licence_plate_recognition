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
img = cv2.imread('/home/imran_afreed/Desktop/licence_plate_recognition/images/two.jpeg')
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
        cv2.imshow("license plate", tmp)

key = cv2.waitKey(0)  
cv2.destroyAllWindows()
