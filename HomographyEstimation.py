#
# Homography Estimation
#
# how to execute this file:
# >> python HomographyEstimation.py --img1 images/1-b0.jpg --img2 images/1-b1.jpg --n 4 --descriptor o
#
import numpy as np 
import cv2 as cv
import argparse

from src.correspondences_estimation import correspondences_estimation
from src.image_processing import img_processing
from src.ransac import ransac

def parse_args():
    parser = argparse.ArgumentParser(description="Problem 1: Homography Estimation")
    parser.add_argument("--img1", type=str, default="images/1-b0.jpg", help="Enter your 1st image path. ")
    parser.add_argument("--img2", type=str, default="images/1-b1.jpg", help="Enter your 2nd image path. ")
    parser.add_argument("--descriptor", type=str, default='o', help="Enter 's' for SIFT; and 'o' for ORB. ")
    parser.add_argument("--n", type=int, default=4, help="Enter n for n-correspondences. ")
    return parser.parse_args() # Parse the argument

def main():

	maxInliers_thresh = 0.9

	img_class = img_processing()
	args = parse_args() # Parse the argument
	ransac_alg = ransac(maxInliers_thresh=maxInliers_thresh, NumberOfInterestPoints=args.n)
	img_class.load2images(args.img1, args.img2)
	# img_class.stitch2images(True)
	correspondence_alg = correspondences_estimation(img_class.img1, img_class.img2, args.descriptor) # An algorithm to find correspondences

	# find features and keypoints
	if img_class.img1 is not None and img_class.img2 is not None:

		# Feature Matching
		correspondence_alg.get_pairs()
		# img_class.drawFeatures(correspondence_alg.matches, correspondence_alg.keypoint_1, correspondence_alg.keypoint_2, img_path=correspondence_alg.img_path_matching)
		img_class.drawMatches(correspondence_alg.matches, correspondence_alg.keypoint_1, correspondence_alg.keypoint_2, 
		 					img_path=correspondence_alg.img_path_matching)

		# run ransac algorithm and estimate homography

		# finalH, finalCorr, inliers = ransac_alg.run(correspondence_alg.correspondences, 'DLT') # direct linear transform (DLT)
		finalH, finalCorr, inliers = ransac_alg.run(correspondence_alg.correspondences, 'NDLT') # normalized DLT

		# Draw matches considering inliers
		img_class.drawMatches(correspondence_alg.matches, correspondence_alg.keypoint_1, correspondence_alg.keypoint_2, 
										img_path='results/match_random_NDLT.png', inliers=finalCorr, drawOutliers=False)
		img_class.drawMatches(correspondence_alg.matches, correspondence_alg.keypoint_1, correspondence_alg.keypoint_2, 
										img_path='results/match_inliers_outliers_NDLT.png', inliers=inliers, drawOutliers=True)
		img_class.drawMatches(correspondence_alg.matches, correspondence_alg.keypoint_1, correspondence_alg.keypoint_2, 
										img_path='results/match_inliers_NDLT.png', inliers=inliers, drawOutliers=False)


if __name__ == '__main__':
	main()



