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

# adaptive_threshold = cv2.bitwise_not(adaptive_threshold)
# kernel = np.ones((2,2),np.uint8)
# adaptive_threshold = cv2.erode(adaptive_threshold,kernel,iterations = 1)
# adaptive_threshold = cv2.bitwise_not(adaptive_threshold)


count,_ = cv2.findContours(adaptive_threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
count_img = cv2.drawContours(adaptive_threshold,count,-1,255,2)
# cv2.imshow(" all counts", count_img)


kernel = np.ones((1,1),np.uint8)
adaptive_threshold = cv2.erode(adaptive_threshold,kernel,iterations = 1)


c  = max(count, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)
# count_img = cv2.rectangle(count_img,(x,y),(x+w,y+h),(0,255,0),5)
count_img = count_img[y+1:y+h-1,x+1:x+w-1]
cv2.imshow("big count", count_img)


config = "--psm 3"
count_img = cv2.resize(count_img, None, fx = 0.4, fy = 0.3)
text = pytesseract.image_to_string(count_img, config=config, lang="eng")
print("licence plate number detected is " + text)

cv2.imshow("gray", gray)
cv2.imshow("adaptive th", adaptive_threshold)
cv2.imwrite("adaptive.png", adaptive_threshold)

key = cv2.waitKey(0)  
cv2.destroyAllWindows()