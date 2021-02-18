import cv2
import numpy as np
import glob
import random
import imutils
import pytesseract
from PIL import Image



crp = cv2.imread("cropped.png")

crp = cv2.resize(crp, None, fx=0.5, fy=0.4)

gray = cv2.cvtColor(crp,cv2.COLOR_BGR2GRAY)

adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 0)

count,_ = cv2.findContours(adaptive_threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
count_img = cv2.drawContours(adaptive_threshold,count,-1,255,2)

# count_img = adaptive_threshold
c  = max(count, key = cv2.contourArea)

x,y,w,h = cv2.boundingRect(c)

gray = gray[y+1:y+h-1,x+1:x+w-1]


config = "--psm 3"
# gray = cv2.resize(gray, None, fx = 0.5, fy = 0.6)
text = pytesseract.image_to_string(gray, config=config, lang="eng")
print("licence plate number detected is " + text)

cv2.imwrite("gray.png", gray)

key = cv2.waitKey(0)  
cv2.destroyAllWindows()