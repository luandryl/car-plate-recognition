import sys
import numpy as np 
import cv2 as cv
import imutils


image = cv.imread("./car_img_repo/car_w.jpg")
image = imutils.resize(image, width=500)

image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray = cv.bilateralFilter(image_gray, 11, 17, 17)
edged = cv.Canny(image_gray, 30, 200)

cv.imshow("Gray", edged)

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

print(screenCnt)



cv.drawContours(image, [screenCnt], -1, (0,255,0), 3)
cv.imshow("C", image)


#mask = np.zeros(image_gray, np.uint8)
#new_img = cv.drawContours(mask, [screenCnt], 0, 255, -1)
#new_img = cv.bitwise_and(image, img, mask=mask)
#cv.imshow("Crop", new_img)

# image_gray = cv.bilateralFilter(image_gray, 11, 17, 17)
# cv.imshow("Remove noise", image_gray)
# edged = cv.Canny(image_gray, 170, 200)
# cv.imshow("Edges", edged)

cv.waitKey(0)