import cv2
import numpy as np
import glob
import random
import imutils
import pytesseract
from PIL import Image


config = "--psm 3"

crp = cv2.imread("cropped.png")
gray = cv2.cvtColor(crp,cv2.COLOR_BGR2GRAY)

im = gray
cv2.imshow("no scaling gray",im)
# cv2.imwrite("no_scaling_gray.png",im)
text = pytesseract.image_to_string(im, config=config, lang="eng")
cv2.imwrite("no_scaling_gray"+text+".png",im)
print("no scaling gray " + text)

im = cv2.resize(gray, None, fx = 0.5, fy = 0.4)
cv2.imshow("0.5 0.4 scaling gray",im)
text = pytesseract.image_to_string(im, config=config, lang="eng")
cv2.imwrite("0.5 0.4 scaling gray"+text+".png",im)
print("0.5 0.4 scaling gray " + text)

im = cv2.resize(gray, None, fx = 0.3, fy = 0.2)
cv2.imshow("0.3 0.2 scaling gray",im)
text = pytesseract.image_to_string(im, config=config, lang="eng")
cv2.imwrite("0.3 0.2 scaling gray"+text+".png",im)
print("0.3 0.2 scaling gray " + text)


adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 0)

im1 = adaptive_threshold
cv2.imshow("no scaling adaptive",im1)
text = pytesseract.image_to_string(im1, config=config, lang="eng")
cv2.imwrite("no scaling adaptive"+text+".png",im1)
print("no scaling adaptive " + text)

im = cv2.resize(adaptive_threshold, None, fx = 0.5, fy = 0.4)
cv2.imshow("0.5 0.4 scaling adaptive",im)
text = pytesseract.image_to_string(im, config=config, lang="eng")
cv2.imwrite("0.5 0.4 scaling adaptive"+text+".png",im)
print("0.5 0.4 scaling adaptive " + text)

im = cv2.resize(adaptive_threshold, None, fx = 0.3, fy = 0.2)
cv2.imshow("0.3 0.2 scaling adaptive",im)
text = pytesseract.image_to_string(im, config=config, lang="eng")
cv2.imwrite("0.3 0.2 scaling adaptive"+text+".png",im)
print("0.3 0.2 scaling adaptive " + text)



count,_ = cv2.findContours(im1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
count_img = cv2.drawContours(im1,count,-1,(255,0,0),2)
cv2.imshow(" all counts", count_img)


c  = max(count, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)
count_img = cv2.rectangle(im1,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("big count bb", count_img)
cv2.imwrite("identify largest possible countur.png",count_img)

count_img = count_img[y+2:y+h-1,x+9:x+w-4]
cv2.imwrite("edges cropped.png",count_img)


# count_img = cv2.bitwise_not(count_img)
kernel = np.ones((1,1),np.uint8)
count_img = cv2.erode(count_img,kernel,iterations = 1)
# count_img = cv2.bitwise_not(count_img)
cv2.imshow("size increaased count_img", count_img)
cv2.imwrite("font size increased.png",count_img)

# count_img = cv2.resize(count_img,None, fx = 1.1, fy = 1)
cv2.imshow("bb cropped no scale ", count_img)
text = pytesseract.image_to_string(count_img, config=config, lang="eng")
cv2.imwrite("no scale font increased border -r"+text+".png",count_img)
print("bb cropped no scale" + text)


imf1 = cv2.resize(count_img,None, fx = 1.4, fy = 1.2)
cv2.imshow("bb cropped 1.4 and 1.2 scale ", imf1)
text = pytesseract.image_to_string(imf1, config=config, lang="eng")
cv2.imwrite("1.4 and 1.2 size font increased bordered -r"+text+".png",count_img)
print("bb cropped 1.4 and 1.2 scale " + text)

imf2 = cv2.resize(count_img,None, fx = 0.3, fy = 0.2)
cv2.imshow("bb cropped 0.3 and 0.2 scale ", imf2)
text = pytesseract.image_to_string(imf2, config=config, lang="eng")
cv2.imwrite("0.3 and 0.2 size font increased bordered -r"+text+".png",count_img)
print("bb cropped 0.3 and 0.2 scale " + text)


key = cv2.waitKey(0)  
cv2.destroyAllWindows()
