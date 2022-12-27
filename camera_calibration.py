############################
#                          #
# Camera Calibration       #
#                          #
# Author: David Wang       #
# Created on Nov. 14, 2022 #
#                          #
# modified from TA's code  #
# in 3DCV class (Fall2022) #
#                          #
############################

import argparse
from src.camera_calibration_video import Calibrator
from src.camera_pose_visualization import camera_pose_visualization

def parse_args():
    parser = argparse.ArgumentParser(description="camera calibration from a video")
    parser.add_argument('--input', 
    					type=str,
    					default='video/calib_video.avi',
                        help='input video for calibration')
    parser.add_argument('--output',
    					type=str,
                        default='results/camera_parameters.npy',
                        help='npy file of camera parameters')
    parser.add_argument('--w',
                        type=int,
                        default=6,
                        help='the width of inner corners of chessboard')
    parser.add_argument('--h',
                        type=int,
                        default=8,
                        help='the height of inner corners of chessboard')
    parser.add_argument('--show',
    					default=True,
                        action='store_true',
                        help='to show the 3D visualization of calibration (require install open3D and only work on Linux)')
    return parser.parse_args() # Parse the argument


if __name__ == '__main__':
    
	args = parse_args() # Parse the argument

    # calibration
	calibrator = Calibrator(args)
	calibrator.run_video()

    # visulize your calibration process
	open3d_visualizer = camera_pose_visualization(calibrator.camera_params_set, [calibrator.inner_w, calibrator.inner_h])
	open3d_visualizer.show_camera_relative_poses()
