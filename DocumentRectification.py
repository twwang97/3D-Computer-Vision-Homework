#
# Document Rectification
#
# how to execute this file:
# >> python DocumentRectification.py --img1 images/book1.jpg 
#

import numpy as np
import cv2 as cv
import argparse

from src.homography_estimation import homography_estimation
from src.image_processing import img_processing
from src.mouse_gui_interface import * 

def parse_args():
	parser = argparse.ArgumentParser(description="Problem 2: Document Rectification")
	parser.add_argument("--img1", type=str, default="images/book1.jpg", help="Enter your image file path. ")
	return parser.parse_args() # Parse the argument

def main():
	args = parse_args() # Parse the argument
	img_class = img_processing()
	img_class.load_image(args.img1) # Read the file
	cv.imwrite('results/original_book.png', img_class.img1)

	points_set_input, img_class.img1 = track_mouse_4clicks(args.img1)
	# points_set_input = np.array([[121, 41], [10, 494], [427, 558], [433, 52]])
	# show_img_in_a_while(img_class.img1)

	points_set_output = np.array([[0, 0], [0, img_class.img1.shape[0]-1], 
		[img_class.img1.shape[1]-1, img_class.img1.shape[0]-1], [img_class.img1.shape[1]-1, 0]])
	correspondences = np.matrix(np.concatenate((points_set_input, points_set_output), axis=1), dtype=float)
	print("4-correspondence coordinate:\n", correspondences)

	camera_mtx = homography_estimation()
	# camera_mtx.DLT(correspondences) # direct linear transform (DLT)
	camera_mtx.normalized_DLT(correspondences) # normalized DLT
	print("homography\n", camera_mtx.H)

	# Points are ordered; points1 corresponds to points2
	transformedImage = img_class.transformImage(img_class.img1.shape[1], img_class.img1.shape[0], img_class.img1, camera_mtx.H)
	cv.imwrite('results/rectified_book.png', transformedImage)

if __name__ == '__main__':
	main()
	
	
