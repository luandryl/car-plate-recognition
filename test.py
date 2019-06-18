import cv2 as cv
import numpy as np
import imutils
import pytesseract

image = cv.imread('./car_img_repo/IMG_20190618_161149701.jpg')
image = imutils.resize(image, width=500)
        
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray = cv.bilateralFilter(image_gray, 11, 17, 17)
edged = cv.Canny(image_gray, 30, 200)

# cv.imshow("Edged", edged)

cnts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv.contourArea, reverse=True)[:10]
screenCnt = None

for c in cnts:
        perimeter = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.018 * perimeter, True)
        if (len(approx) == 4):
                screenCnt = approx
                break

mask = np.zeros(image_gray.shape,np.uint8)
new_image = cv.drawContours(image,[screenCnt],0,255,-1,)
# cv.imshow('olar', new_image)

mask = np.zeros(image_gray.shape, np.uint8)
new_img = cv.drawContours(mask, [screenCnt], 0, 255, -1)
new_image = cv.bitwise_and(image,image, mask=mask)

# cv.imshow("Contours", new_img)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
licence_plate = image_gray[topx:bottomx+1, topy:bottomy+1]

# licence_plate = cv.medianBlur(licence_plate, 5);
# licence_plate = cv.adaptiveThreshold(licence_plate, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 43,2);

cv.imshow("plate", licence_plate)

import pytesseract
text = pytesseract.image_to_string(licence_plate)
print(text)

'''
_,contours = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
w,h,x,y = 0,0,0,0

for contour in contours:
    area = cv.contourArea(contour)

    if area > 6000 and area < 4000:
        [x,y,w,h] = cv.boundingRect(contour)

print([x,y,w,h])
'''
# cv.imshow("Img", edged)

cv.waitKey(0)
