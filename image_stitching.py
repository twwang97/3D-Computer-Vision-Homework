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

# Python Usage
# python3 image_stitching.py 
# python3 image_stitching.py --imgDir successive_images/scottsdale --outputDir results 

from src.utils_stitching import *

import argparse

def parse_args():
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgDir", type=str, default="successive_images/hills",  # successive_images/scottsdale
                    help="path to input directory of images to stitch")
    parser.add_argument("--outputDir", type=str, default="results", 
                    help="path to the output image")
    args = vars(parser.parse_args())
    return args

def main():

    args = parse_args() # Parse the argument
    images = load_dataset(args["imgDir"])
    status, stitchedImg = stitch_images(images, args["outputDir"])

    # if the status is '0', then OpenCV successfully performed image
    # stitching
    if status == 0:
        # crop out the largest rectangular
        # region from the stitched image
        crop_image(stitchedImg, args["outputDir"])

    # otherwise the stitching failed, likely due to not enough keypoints)
    # being detected
    else:
        print("[INFO] image stitching failed ({})".format(status))


if __name__ == '__main__':
    main()