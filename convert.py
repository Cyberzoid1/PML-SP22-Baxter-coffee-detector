""" This script pre processes images to find the smaller image of interest
Take the image, look for a square the represents the coffeemaker screen
Output that smaller image to a new matching directory
"""

import os
from time import sleep
import cv2
import numpy as np


def find_square(image):
    """ Load image, grayscale, median blur, sharpen image """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold and morph close
    thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]            # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and filter using threshold area
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 5000
    max_area = 10000
    image_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        print("Area: %s Image: %d" % (area, image_number))
        if area > 100: #min_area and area < max_area:
            # print("Match Area: %s Image: %d" % (area, image_number))
            x,y,w,h = cv2.boundingRect(c)
            ratio = float(w) / float(h)
            print("w: %d, h: %d, Ratio: %f" % (w, h, ratio))
            if ratio > 1.0 and ratio < 1.3:
                print("good ratio")
                ROI = image[y:y+h, x:x+w]
                # cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
                # cv2.imshow('ROI_{}.png'.format(image_number), ROI)
                # cv2.waitKey(0)
                # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                return ROI
            # else:
            #     print("bad ratio")
            image_number += 1

    # cv2.imshow('close', close)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('gray', gray)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    return None

def main():
    IMG_SOURCE_DIR="images"
    IMG_DEST_DIR="images-sm"
    if not os.path.exists(IMG_DEST_DIR):
        os.makedirs(IMG_DEST_DIR)

    directory_labels_list = os.listdir(IMG_SOURCE_DIR)
    print(directory_labels_list)
    directory_labels_list.sort()

    for dir in directory_labels_list:
        path = IMG_SOURCE_DIR + "/" + dir
        print("Current directory: %s" % path)
        images_list = os.listdir(path)

        for image_raw in images_list:
            print("Converting %s" % image_raw)
            image_path = path + "/" + image_raw
            image = cv2.imread(image_path)
            # cv2.imshow(image_raw, image)
            target_image = find_square(image)       # Find and return image of screen
            if target_image is not None:
                # cv2.imshow(image_raw, target_image)

                if not os.path.exists(IMG_DEST_DIR + "/" + dir):
                    os.makedirs(IMG_DEST_DIR + "/" + dir)

                write_path = IMG_DEST_DIR + "/" + dir + "/sm-" + image_raw
                print("Writing to %s" % write_path)
                cv2.imwrite(write_path, target_image)
            else:
                print("ERROR: Did not find square")


if __name__ == '__main__':
    main()
