import cv2
from PIL import Image
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
image = cv2.imread(r"C:\Users\bala\Downloads\h.jpg")

dilated_img = cv2.dilate(image[:,:,1], np.ones((7, 7), np.uint8))
bg_img = cv2.medianBlur(dilated_img, 21)
diff_img = 255 - cv2.absdiff(image[:,:,1], bg_img)
norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imshow('th', cv2.resize(th, (0, 0), fx = 0.5, fy = 0.5))
text = pytesseract.image_to_string(th)
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()