############################
#                          #
#     Image Stitching      #
#                          #
# Author: David Wang       #
# Created on Dec. 27, 2022 #
#                          #
# modified from            #
# apoorva-dave's github    #
#                          #
############################

# Purpose:
## Creates a panorama from given set of images.

# reference 
## https://github.com/apoorva-dave/Image-Stitching

import glob # to import image paths
import numpy as np
import imutils
import cv2

def stitch_images(images, imgOutputDir):
    # initialize OpenCV's image stitcher object 
    # and then perform the image stitching
    print("[INFO] stitching images...")
    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)

    imgOutputPath = imgOutputDir + '/stitching.png'
    display_save_image(stitched, imgOutputPath)

    return status, stitched

def load_dataset(img_dir):
    # grab the paths to the input images and initialize our images list
    print("[INFO] loading images...")
    img_path_pattern = img_dir + "/*.jpg"
    imagePaths = [f for f in glob.glob(img_path_pattern)]
    print(imagePaths)
    images = []

    # loop over the image paths, load each one, and add them to our
    # images to stitch list
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        images.append(image)
    return images

def display_save_image(img, imgOutputPath):

    # write the output stitched image to disk
    cv2.imwrite(imgOutputPath, img)
    # display the output stitched image to our screen
    cv2.imshow(imgOutputPath, img)
    cv2.waitKey(0)
    cv2.destroyWindow(imgOutputPath)

def crop_image(stitchedImg, imgOutputDir):
    imgOutputPath = imgOutputDir + '/cropped.png'
    # create a 10 pixel border surrounding the stitched image
    print("[INFO] cropping...")
    stitchedImg = cv2.copyMakeBorder(stitchedImg, 10, 10, 10, 10,
                                  cv2.BORDER_CONSTANT, (0, 0, 0))

    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    gray = cv2.cvtColor(stitchedImg, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # find all external contours in the threshold image then find
    # the *largest* contour which will be the contour/outline of
    # the stitched image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # allocate memory for the mask which will contain the
    # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # create two copies of the mask: one to serve as our actual
    # minimum rectangular region and another to serve as a counter
    # for how many pixels need to be removed to form the minimum
    # rectangular region
    minRect = mask.copy()
    sub = mask.copy()

    # keep looping until there are no non-zero pixels left in the
    # subtracted image
    while cv2.countNonZero(sub) > 0:
        # erode the minimum rectangular mask and then subtract
        # the thresholded image from the minimum rectangular mask
        # so we can count if there are any non-zero pixels left
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    # find contours in the minimum rectangular mask and then
    # extract the bounding box (x, y)-coordinates
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

    # use the bounding box coordinates to extract the our final
    # stitched image
    stitchedImg = stitchedImg[y:y + h, x:x + w]

    display_save_image(stitchedImg, imgOutputPath)