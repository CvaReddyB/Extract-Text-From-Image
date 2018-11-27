from PIL import Image
import cv2
import pytesseract
import numpy as np


pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
image = cv2.imread(r'C:\Users\bala\Downloads\d.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

retval, threshold = cv2.threshold(gray_image, 170, 255,cv2.THRESH_BINARY)
th3 = cv2.adaptiveThreshold(gray_image,200,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,12)

text= pytesseract.image_to_string(th3)
print(text)

cv2.imwrite('./gray.jpg',th3)
cv2.imshow('threshold image',th3)

cv2.waitKey(0)
cv2.destroyAllWindows()




























'''
th2 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
'''
'''
kernel = np.ones((1, 1), np.uint8)
img = cv2.dilate(gray_image, kernel, iterations=1)
img = cv2.erode(gray_image, kernel, iterations=1)
'''
#bg_img = cv2.medianBlur(gray_image, 21)
