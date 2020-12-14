# Bounding boxes for infected apples
# Authors: George Truman, Harry Dodwell, Christian Prather
#

import cv2
import sys
import numpy as np
from detect import run_inference
import os
import pathlib

current_dir = os.getcwd()
# import matplotlib
# from matplotlib import pyplot as plt


# Points is a list of lists. Each element contains and x and a y coordinate
# Text contains the type of rot in the image
# Image is the image that the points are marked on and the text is placed
def draw_markers(points, text, image):

    # Mark the points with rot
    for p in points:
        # TODO change this to work with a rectangle bounding box
        # cv2.drawMarker(image, (int(p.pt[0]), int(p.pt[1])), (100, 255, 0), thickness= 2)
        # cv2.rectangle(image, ((int(p.pt[0]) - (int(p.size)/2)), (int(p.pt[1]) - (int(p.size)/2))), 
        #                      ((int(p.pt[0]) + (int(p.size)/2)), (int(p.pt[1]) + (int(p.size)/2))),
        #                      (255, 50,50), 3)
        radius = int(p.size/2)
        cv2.rectangle(image, (int(p.pt[0] - radius)  , int(p.pt[1] - radius) ), ( int(p.pt[0] + radius), int(p.pt[1] + radius) ), (50,50, 255), 2)
    # Put the label in the top left corner
    cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(50,50, 255), thickness=2)

    cv2.imshow("modified", image)
    cv2.waitKey(0)
    cv2.imwrite("BestRustSeg.jpg", image)

    return image


def denoise_image(input_image):
    min_thresholds = [42,16,25]
    max_thresholds = [255,255,255]
    # bgr_img = cv2.imread("test_demo_rot.JPG")  # Get query image
    bgr_img = cv2.imread(input_image)
    # cv2.imshow("Source", bgr_img)
    
    threshold_image = np.full((bgr_img.shape[0], bgr_img.shape[1]), 255, dtype=np.uint8)
    kernel = np.ones((3,3),np.uint8)

    hsv_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    splits = cv2.split(hsv_image)

    for i in range(3):
        _, low_img = cv2.threshold(splits[i], min_thresholds[i], 255, cv2.THRESH_BINARY)
        _, high_img = cv2.threshold(splits[i], max_thresholds[i], 255, cv2.THRESH_BINARY_INV)

        threshold_combo = cv2.bitwise_and(low_img, high_img)
        threshold_image = cv2.bitwise_and(threshold_image, threshold_combo)

    morph = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Threshold", threshold_image)
    cv2.imshow("Morph", morph)
    cv2.imwrite("BestRustMorph.jpg", morph)
    cv2.imwrite("BestRustThresh.jpg", threshold_image)

 


    # _, _, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=4)
    # print(centroids)


    test_image = cv2.bitwise_or(bgr_img, bgr_img, mask= morph)
    # cv2.imshow("Areas of interest", test_image)

    # Blob detection
    params = cv2.SimpleBlobDetector_Params()
    # Area
    params.filterByArea = True
    params.minArea = 100.0
    params.maxArea = 100000000.0

    params.minThreshold = 200
    params.maxThreshold = 255

    params.filterByConvexity = True
    params.minConvexity = 0.66

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(morph)
    # for point in keypoints:
    #     print(point.pt)
    #     print("Size",point.size)    


    cv2.waitKey(0)

    # return bgr_img, centroids
    return bgr_img, keypoints


if __name__ == "__main__":
    input_image = sys.argv[1]
    # input_image = pathlib.Path(current_dir + input_image)

    class_name = run_inference(input_image)
    dst, points = denoise_image(input_image)
    # dst = denoise_image()

    # p = [[100, 100], [200, 200]]
    # t = ['Point 1', 'Point 2']
    # class_name = "TEST"
    draw_markers(points, class_name, dst)
    print(class_name)
