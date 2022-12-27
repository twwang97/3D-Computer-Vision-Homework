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

import argparse
from src.recification_function import * 


# INPUT_IMG1_PATH = 'bike1.png'
# INPUT_IMG2_PATH = 'bike2.png'
# INPUT_IMG1_PATH = 'sofa1.png'
# INPUT_IMG2_PATH = 'sofa2.png'
# INPUT_IMG1_PATH = 'umbrella1.png'
# INPUT_IMG2_PATH = 'umbrella2.png'
OUTPUT_ORIGIN_PATH = 'results/original2images.png'
OUTPUT_EPILINE_PATH = 'results/epilines2images.png'
OUTPUT_RECTIFICATION_PATH = 'results/rectified2images.png'

def parse_args():
    parser = argparse.ArgumentParser(description="Stereo Rectification")
    parser.add_argument("--img1", type=str, default='images/bike1.png', help="Enter your 1st image path. ")
    parser.add_argument("--img2", type=str, default='images/bike2.png', help="Enter your 2nd image path. ")
    
    # Parse the argument
    args = parser.parse_args()
    return args
def main():

	args = parse_args()
	###############################
	#
	# Match keypoints of 2 images
	#
	###############################

	img1, img2 = read_resize_2images(args.img1, args.img2, OUTPUT_ORIGIN_PATH)

	# Detect keypoints and their descriptors
	kp1, kp2, des1, des2 = find_sift_keypoints(img1, img2, Visualization=False)
	# Match keypoints in both images
	kp1, kp2 = matching_keypoints(img1, img2, kp1, kp2, des1, des2, Visualization=False)

	###############################
	#
	# Stereo rectification
	#
	###############################

	# Calculate the fundamental matrix for the cameras and return only inlier keypoints
	fundamental_matrix, kp1, kp2= find_F_and_inliers(kp1, kp2)

	# Find epilines
	# Find epilines corresponding to points in right image (second image) 
	lines2 = find_epilines(kp2, fundamental_matrix)
	# Find epilines corresponding to points in left image (first image) 
	lines1 = find_epilines(kp1, fundamental_matrix)
	# Draw epilines on corresponding two images
	show_epilines_2images(img1, img2, lines1, lines2, kp1, kp2, outputPath=OUTPUT_EPILINE_PATH, showImg=False)

	# Stereo rectification (uncalibrated variant)
	# Adapted from: https://stackoverflow.com/a/62607343
	_, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(kp1), np.float32(kp2), fundamental_matrix, imgSize=(img1.shape[1], img1.shape[0]))

	# Undistort (rectify) the images and save them
	# Adapted from: https://stackoverflow.com/a/62607343
	img1_rectified = cv.warpPerspective(img1, H1, (img1.shape[1], img1.shape[0]))
	img2_rectified = cv.warpPerspective(img2, H2, (img2.shape[1], img2.shape[0]))
	show_rectified_2images(img1_rectified, img2_rectified, outputPath=OUTPUT_RECTIFICATION_PATH, showImg=False)


if __name__ == '__main__':
	main()