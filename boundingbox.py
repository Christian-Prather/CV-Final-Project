# Bounding boxes for infected apples
# Authors: George Truman, Harry Dodwell, Christian Prather
#

import cv2
import sys
import numpy as np
# import matplotlib
# from matplotlib import pyplot as plt


# Points is a list of lists. Each element contains and x and a y coordinate
# Text contains the type of rot in the image
# Image is the image that the points are marked on and the text is placed
def draw_markers(points, text, image):

    # Mark the points with rot
    for p in points:
        # TODO change this to work with a rectangle bounding box
        cv2.drawMarker(image, (int(p[0]), int(p[1])), (255, 0, 0))

    # Put the label in the top left corner
    cv2.putText(image, text, (40, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, color=(255, 0, 0))

    cv2.imshow("modified", image)
    cv2.waitKey(0)

    return image


def denoise_image():
    min_thresholds = [42,16,25]
    max_thresholds = [255,255,255]
    bgr_img = cv2.imread("test_demo_rot.JPG")  # Get query image
    cv2.imshow("Source", bgr_img)
    
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

    # Blob detection
    # params = cv2.SimpleBlobDetector_Params()
    # # Area
    # params.filterByArea = True
    # params.minArea = 100.0
    # params.maxArea = 100000000.0

    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(morph)
    # for point in keypoints:
    #     print(point.pt)    


    # _, _, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=4)
    # print(centroids)


    # denoise the image
    # dst = cv2.fastNlMeansDenoisingColored(bgr_img,None,10,10,7,21)
    # cv2.imshow("denoise", dst)
    # dst = bgr_img
    # # convert clean photo to 
    # gray_img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray", gray_img)
    # ret, thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow("Thresh", thresh)
    # cv2.waitKey(0)

    # # noise removal
    # kernel = np.ones((3,3),np.uint8)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
   
   
   
    # sure_bg = cv2.dilate(morph,kernel,iterations=3)

    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(morph,cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg,sure_fg)

    # # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers+1
    # # Now, mark the region of unknown with zero
    # markers[unknown==255] = 0

    # # watershed for 
    # markers = cv2.watershed(bgr_img,markers)
    # # dst = bgr_img
    # bgr_img[markers == -1] = [255,0,0]

    
    # # # cv2.imshow("markers", markers)
    # cv2.imshow("img", bgr_img)
    # # cv2.waitKey(0)
    # # bgr_img = cv2.imread("apple_BR.jpg")  # Get query image
    # # cv2.imshow("img", bgr_img)
    # # cv2.waitKey(0)
    # # convert to gray
    # # gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    # # dst = cv2.fastNlMeansDenoisingColored(bgr_img, None, 10, 10, 7, 21)
    # # cv2.imshow("denoise", dst)
    # # ret, thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # # cv2.imshow("img", thresh)


    # morph = cv2.resize(morph, (bgr_img.shape[1], bgr_img.shape[0]))
    test_image = cv2.bitwise_or(bgr_img, bgr_img, mask= morph)
    cv2.imshow("Test", test_image)
    cv2.waitKey(0)

    # return bgr_img, centroids
    return bgr_img


if __name__ == "__main__":
    # dst, points = denoise_image()
    dst = denoise_image()

    # p = [[100, 100], [200, 200]]
    # t = ['Point 1', 'Point 2']
    class_name = "TEST"
    # draw_markers(points, class_name, dst)
