import sys
import numpy as np 
import cv2 as cv
import imutils
import pytesseract
import os


def recog(car):
        image = cv.imread(car)
        image = imutils.resize(image, width=500)

        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_gray = cv.bilateralFilter(image_gray, 11, 17, 17)
        edged = cv.Canny(image_gray, 30, 200)
        
        # cv.imshow("Gray", edged)
        
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
        # print(screenCnt)

        mask = np.zeros(image_gray.shape, np.uint8)
        new_img = cv.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv.bitwise_and(image,image, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = image_gray[topx:bottomx+1, topy:bottomy+1]
        
        cv.imshow("Olar", Cropped)
        
        plate_array = []
        text = pytesseract.image_to_string(Cropped, config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        plate_array.append(text)
        
        k = cv.waitKey(27)
        if (k == 27):
                cv.destroyAllWindows()
        
        return plate_array


path = './car_img_repo/'
files = os.listdir(path)
licence_plates = []

for index, file in enumerate(files): 
        string_img = "./car_img_repo/" + file
        print(string_img)
        licence_plates.append(recog(string_img))
        
print(licence_plates)
