############################
#                          #
#    Visual Odometry       #
#                          #
# Author: David Wang       #
# Created on Nov. 14, 2022 #
#                          #
# modified from PYSLAM.    #
# Please refer to          #
# luigifreda's github      #
#                          #
############################

# modified from PYSLAM
# https://github.com/luigifreda/pyslam

from vo.src.visual_odometry import VisualOdometry
from vo.src.camera  import PinholeCamera
from vo.src.dataset import dataset_factory

from vo.src.feature.feature_tracker_menu import feature_tracker_selector
from vo.src.utils.utils_viewer import viewer_set

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="visual odometry")
    parser.add_argument('--imgDir', 
    					type=str,
    					default='input_vo_frames',
                        help='Your directory for the input images')
    parser.add_argument('--camParams',
    					type=str,
                        default='results_calibration/realsenseD435i_camera_params.npy',
                        help='npy file for camera intrinsic matrix')
    parser.add_argument('--features',
    					type=int,
                        default=1000,
                        help='Number of features to be tracked')
    parser.add_argument('--trackerConfig',
                        type=str,
                        default='LK_FAST',
                        help='your tracker configuration: orb, brisk, sift, sift_root, akaze, LK_SHI_TOMASI, LK_FAST')
    parser.add_argument('--poseMethod',
                        type=str,
                        default='opencv',
                        help='pose estimated from opencv or mymethod')
    parser.add_argument('--show',
    					type=bool,
    					default=True,
                        help='to show the 3D visualization of calibration (require install open3D and only work on Linux)')
    return parser.parse_args() # Parse the argument

if __name__ == "__main__":	

	args = parse_args() # Parse the argument

	imgPath = '*.png'
	imgDir = args.imgDir # 'frames'
	dataset = dataset_factory(imgDir, imgPath)

	camera_width = 640
	camera_height = 360
	camera_params_npy = args.camParams # 'camera_params.npy'
	cam = PinholeCamera(camera_width, camera_height, camera_params_npy)

	# select your tracker configuration (see the file feature_tracker_configs.py) 
	# LK_SHI_TOMASI, LK_FAST
	# SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT
	feature_tracker = feature_tracker_selector(args.trackerConfig, args.features)

	# create visual odometry object 
	vo = VisualOdometry(cam, feature_tracker, is_transformed_grayscale=False, UsePoseNewMethod=args.poseMethod)

	## Drawing parameters
	UseOpen3D = True
	is_draw_traj_img = False # True
	is_draw_3d = args.show # True
	is_draw_matched_points_count = args.show # True 
	viewers = viewer_set(UseOpen3D, is_draw_3d, is_draw_matched_points_count, is_draw_traj_img)
	viewers.init()

	#################


	img_id = 0
	while dataset.isOk():

		img = dataset.getImage(img_id)

		if img is not None:
			# main visual odometry function 
			vo.track(img, img_id)  

			#  start drawing from the third image
			stop_flag = viewers.visualize_process(img_id, vo)

		if stop_flag: # press 'q' to exit!
			break

		img_id += 1

	viewers.destroy_all_viewers()
