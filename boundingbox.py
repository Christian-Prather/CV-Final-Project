# Bounding boxes for infected apples
# Authors: George Truman, Harry Dodwell, Christian Prather
#

# Libraries needed for inference and cv processing
import cv2
import sys
import numpy as np
# Tensorflow
from detect import run_inference
import os
import pathlib

# For referencing image paths
current_dir = os.getcwd()

# Points is a list of lists. Each element contains and x and a y coordinate
# Text contains the type of rot in the image
# Image is the image that the points are marked on and the text is placed
# Returns: Final AR image
def draw_markers(points, text, image):

    # Mark the points with rot
    for p in points:
        radius = int(p.size/2)
        cv2.rectangle(image, (int(p.pt[0] - radius)  , int(p.pt[1] - radius) ), ( int(p.pt[0] + radius), int(p.pt[1] + radius) ), (50,50, 255), 2)
    # Put the label in the top left corner
    cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(50,50, 255), thickness=2)

    cv2.imshow("modified", image)
    cv2.waitKey(0)
    # For saving out
    #cv2.imwrite("BestSeg.jpg", image)

    return image

# Primary logic of function
# Functionality:
# 1) Denoise image (optional)
# 2) Threshold and morph
# 3) Blob detection
# returns: morphed image and blob keypoints
def denoise_image(input_image):
    # Values found through exploration using find_morph.py
    min_thresholds = [42,16,25]
    max_thresholds = [255,255,255]
    bgr_img = cv2.imread(input_image)
    # cv2.imshow("Source", bgr_img)

    # Set up black image to build on    
    threshold_image = np.full((bgr_img.shape[0], bgr_img.shape[1]), 255, dtype=np.uint8)
    # 3x3 Kernel for morphing
    kernel = np.ones((3,3),np.uint8)

    # Convert image to HSV 
    hsv_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    splits = cv2.split(hsv_image)

    # For each H,S,V subsection of the image threshold and combine using preset thresholds
    for i in range(3):
        _, low_img = cv2.threshold(splits[i], min_thresholds[i], 255, cv2.THRESH_BINARY)
        _, high_img = cv2.threshold(splits[i], max_thresholds[i], 255, cv2.THRESH_BINARY_INV)

        threshold_combo = cv2.bitwise_and(low_img, high_img)
        threshold_image = cv2.bitwise_and(threshold_image, threshold_combo)

    # Morph threshold combination image using close/open to get better hole filling
    morph = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Display image data
    cv2.imshow("Threshold", threshold_image)
    cv2.imshow("Morph", morph)
    cv2.imwrite("BestRustMorph.jpg", morph)
    cv2.imwrite("BestRustThresh.jpg", threshold_image)

    # Debug tool to show final morph overlayed with source image
    # test_image = cv2.bitwise_or(bgr_img, bgr_img, mask= morph)
    # cv2.imshow("Areas of interest", test_image)

    # Blob detection
    params = cv2.SimpleBlobDetector_Params()
    # Area
    params.filterByArea = True
    params.minArea = 100.0
    # Max out size limit
    params.maxArea = 100000000.0

    # Threshold produced very nice contrasts so dont need large range for blob
    params.minThreshold = 200
    params.maxThreshold = 255

    # Allow for non circular blobs
    params.filterByConvexity = True
    params.minConvexity = 0.66

    # Run detection
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(morph)
    
    # Debugging  
    # for point in keypoints:
    #     print(point.pt)
    #     print("Size",point.size)    

    # Dsiplay images
    cv2.waitKey(0)

    # return bgr_img, centroids
    return bgr_img, keypoints


if __name__ == "__main__":
    # input image is argument on execution (requires full path)
    input_image = sys.argv[1]

    # Run image through detect.py to get classification
    class_name = run_inference(input_image)
    # Morph and threshold image getting blobs keypoints out
    dst, points = denoise_image(input_image)
    # dst = denoise_image()

    # Overlay image with AR bounding boxes
    draw_markers(points, class_name, dst)
    print(class_name)
