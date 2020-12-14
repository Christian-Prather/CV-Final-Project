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
        cv2.drawMarker(image, (p[0], p[1]), (255, 0, 0))

    # Put the label in the top left corner
    cv2.putText(image, text, (40, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, color=(255, 0, 0))

    cv2.imshow("modified", image)
    cv2.waitKey(0)

    return image


def denoise_image():
    
    bgr_img = cv2.imread("test_demo_rot.JPG")  # Get query image

    # denoise the image
    dst = cv2.fastNlMeansDenoisingColored(bgr_img,None,10,10,7,21)
    # cv2.imshow("denoise", dst)

    # convert clean photo to 
    gray_img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("denoise", dst)
    ret, thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow("img", thresh)
    # cv2.waitKey(0)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    cv2.imwrite("WorstMorphRot.jpg", opening)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    # watershed for 
    markers = cv2.watershed(bgr_img,markers)
    dst[markers == -1] = [255,0,0]
    # cv2.imshow("markers", markers)
    cv2.imshow("img", dst)
    cv2.imwrite("worstSegRot.jpg", dst)
    cv2.waitKey(0)
    # bgr_img = cv2.imread("apple_BR.jpg")  # Get query image
    # cv2.imshow("img", bgr_img)
    # cv2.waitKey(0)
    # convert to gray
    # gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    # dst = cv2.fastNlMeansDenoisingColored(bgr_img, None, 10, 10, 7, 21)
    # cv2.imshow("denoise", dst)
    # ret, thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow("img", thresh)
    # cv2.waitKey(0)

    return dst


if __name__ == "__main__":
    dst = denoise_image()

    p = [[100, 100], [200, 200]]
    t = ['Point 1', 'Point 2']

    draw_markers(p, t, dst)
