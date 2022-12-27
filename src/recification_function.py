############################
#                          #
# Stereo Rectification     #
#                          #
# Author: David Wang       #
# Created on Oct. 28, 2022 #
#                          #
############################


# reference
# https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def read_resize_2images(INPUT_IMG1_PATH, INPUT_IMG2_PATH, outputPath=None):

	# Read both images and convert to grayscale
	img1 = cv.imread(INPUT_IMG1_PATH, cv.IMREAD_GRAYSCALE)
	img2 = cv.imread(INPUT_IMG2_PATH, cv.IMREAD_GRAYSCALE)
	img1 = cv.resize(img1, (640,480))
	img2 = cv.resize(img2, (640,480))
	# cv.imshow('img1',img1)
	# cv.waitKey(0)
	# cv.imshow('img2',img2)
	# cv.waitKey(0)
	stitch2images(img1, img2, outputPath=outputPath)
	return img1, img2

def find_sift_keypoints(img1, img2, Visualization=False):
	# Detect keypoints and their descriptors
	# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

	# Initiate SIFT detector
	sift = cv.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	# Visualize keypoints
	if Visualization:
		imgSift = cv.drawKeypoints(img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv.imshow("SIFT Keypoints", imgSift)
		cv.waitKey(0)

	return kp1, kp2, des1, des2

def matching_keypoints(img1, img2, kp1, kp2, des1, des2, Visualization=False):
	# Match keypoints in both images
	# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	# Keep good matches: calculate distinctive image features
	# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
	# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
	matchesMask = [[0, 0] for i in range(len(matches))]
	good = []
	pts1 = []
	pts2 = []

	for i, (m, n) in enumerate(matches):
	    if m.distance < 0.7*n.distance:
	        # Keep this keypoint pair
	        matchesMask[i] = [1, 0]
	        good.append(m)
	        pts2.append(kp2[m.trainIdx].pt)
	        pts1.append(kp1[m.queryIdx].pt)

	# Draw the keypoint matches between both pictures
	# based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
	draw_params = dict(matchColor=(0, 255, 0),
	                   singlePointColor=(255, 0, 0),
	                   matchesMask=matchesMask[300:500],
	                   flags=cv.DrawMatchesFlags_DEFAULT)

	keypoint_matches = cv.drawMatchesKnn(
	    img1, kp1, img2, kp2, matches[300:500], None, **draw_params)
	# cv.imshow("Keypoint matches", keypoint_matches)
	# cv.waitKey(0)

	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)

	return pts1, pts2



def find_F_and_inliers(pts1, pts2): 
    # Calculate the fundamental matrix for the cameras
    # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

    # We select only inlier points
    pts1 = pts1[inliers.ravel() == 1] # ravel: flatten the array
    pts2 = pts2[inliers.ravel() == 1]
    return fundamental_matrix, pts1, pts2

# Visualize epilines
# Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
def drawEpilines_1img(img1src, img2src, lines, pts1src, pts2src):
    # img1 - image on which we draw the epilines for the points in img2
    # lines - corresponding epilines
    r, c = img1src.shape
    img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color

def find_epilines(kp1, fundamental_matrix):
	# Find epilines corresponding to points in the image (second image)
	epilines_1 = cv.computeCorrespondEpilines(kp1.reshape(-1, 1, 2), 2, fundamental_matrix)
	epilines_1 = epilines_1.reshape(-1, 3)
	return epilines_1

def stitch2images(img1, img2, outputPath=False):
    # Stitch 2 images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    img_channels = 3 # rgb data
    if len(img1.shape) == 2: # gray image
        img_channels = 1 # gray image

    out = np.zeros((max([rows1,rows2]), cols1+cols2, 3), dtype='uint8')

    if img_channels == 3: # rgb data
        # Place the first image to the left
        out[:rows1,:cols1,:] = np.dstack([img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]])

        # Place the next image to the right of it
        out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]])

    elif img_channels == 1: # gray data
        # Place the first image to the left
        out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

        # Place the next image to the right of it
        out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])  

    if outputPath:
        cv.imwrite(outputPath, out)

def show_epilines_2images(img1src, img2src, lines1, lines2, pts1src, pts2src, outputPath=None, showImg=False):
	# Draw epilines for the 2nd image
	img2_e, _ = drawEpilines_1img(img1src, img2src, lines2, pts1src, pts2src)
	# Draw epilines for the 1st image
	img1_e, _ = drawEpilines_1img(img2src, img1src, lines1, pts2src, pts1src)

	stitch2images(img2_e, img1_e, outputPath=outputPath)

	if showImg:
		plt.subplot(121), plt.imshow(img2_e)
		plt.subplot(122), plt.imshow(img1_e)
		plt.suptitle("Epilines in the 2nd (left) / 1st (right) image")
		plt.show()

def show_rectified_2images(img1_rectified, img2_rectified, outputPath=None, showImg=False):

	# cv.imwrite("rectified_1.png", img1_rectified)
	# cv.imwrite("rectified_2.png", img2_rectified)
	stitch2images(img1_rectified, img2_rectified, outputPath=outputPath)

	if showImg:
		# Draw the rectified images
		fig, axes = plt.subplots(1, 2, figsize=(15, 10))
		axes[0].imshow(img1_rectified, cmap="gray")
		axes[1].imshow(img2_rectified, cmap="gray")
		axes[0].axhline(250) # Add a horizontal line across the Axes.
		axes[1].axhline(250)
		axes[0].axhline(450)
		axes[1].axhline(450)
		plt.suptitle("Rectified images")
		# plt.savefig("rectified_images.png")
		plt.show()