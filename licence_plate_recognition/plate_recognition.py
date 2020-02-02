import numpy as np
import cv2

def detect_carplate(image):

    # A haar cascade for UK-plates
    plate_cascade_UK = cv2.CascadeClassifier('UKPlates66x20P.xml')

    # A haar cascade for Russian-plates
    plate_cascade_RU = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    # Reading the image, returning a HSV (hue saturation value) for each pixel in the image; saved in img 
    img = cv2.imread(image)
    img_RU = img.copy()

    # Changing color to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_RU_gray = gray.copy()

    # Detecting all the cordinates, which the program detects a UK license plate. It will return an
    # array of (x-coord, y-coord, width, height).
    UK_plates = plate_cascade_UK.detectMultiScale(gray, 1.3, 5)

    RU_plates = plate_cascade_RU.detectMultiScale(gray, 1.3, 5)

    smallest_UK = (0,0,0,0)
    area_UK = float("inf")

    for (x,y,w,h) in UK_plates:
        if area_UK > w*h:
            area_UK = w*h
            smallest_UK = (x,y,w,h)


    smallest_RU = (0,0,0,0)
    area_RU = float("inf")

    for (x,y,w,h) in RU_plates:
        if area_RU > w*h:
            area_RU = w*h
            smallest_RU = (x,y,w,h)


    img = cv2.rectangle(img,(smallest_UK[0],smallest_UK[1]),(smallest_UK[0]+smallest_UK[2],smallest_UK[1]+smallest_UK[3]),(255,0,0),2)

    crop_img = gray[smallest_UK[1]:smallest_UK[1]+smallest_UK[3], smallest_UK[0]:smallest_UK[0]+smallest_UK[2]]


    img_RU = cv2.rectangle(img_RU,(smallest_RU[0],smallest_RU[1]),(smallest_RU[0]+smallest_RU[2],smallest_RU[1]+smallest_RU[3]),(255,0,0),2)

    crop_img_RU = img_RU_gray[smallest_RU[1]:smallest_RU[1]+smallest_RU[3], smallest_RU[0]:smallest_RU[0]+smallest_RU[2]]

    # Later on if needed the unsharp mask can be applied by using these lines of codes
    # DPI should be fixed to 300 if OCR does not recognizes the text
    #gaussian_3 = cv2.GaussianBlur(crop_img, (9,9), 10.0)
    #unsharp_image = cv2.addWeighted(crop_img, 1.5, gaussian_3, -0.5, 0, crop_img)

    try:
        cv2.imwrite('cropped_UK.jpg', crop_img)
    except:
        print('Could not find the plate!')
        pass

    try:
        cv2.imwrite('cropped_RU.jpg', crop_img_RU)
    except:
        print('Could not find the plate!')
        pass

    try:
        cv2.imshow('img_UK',img)
        cv2.imshow('img_RU',img_RU)
        cv2.imshow('cropped_UK',crop_img)
        cv2.imshow('cropped_RU', crop_img_RU)
    except:
        cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_carplate('./car.jpg')