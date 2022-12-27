############################
#                          #
# Camera Pose Estimation   #
# (See Trajectory)         #
#                          #
# 2D-3D correspondence     #
# method: P3P + RANSAC     #
#                          #
# Author: David Wang       #
# Created on Oct. 20, 2022 #
#                          #
############################

from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import open3d as o3d
import argparse

from src.data_loader import data_loader
from src.cameraPose import cameraPose
from src.timeRecording import timeRecording_

def parse_args():
	print('Match 2D-3D Correspondences and Plot the Trajectory')
	parser = argparse.ArgumentParser(description="Match 2D-3D Correspondences and Plot the Trajectory")
	parser.add_argument("--onlyshow", type=int, default=0, help="Show the trajectory without running pnp, then enter 1. ")
	# pnp option: ['opencv_PnPRansac', 'p3p_Grunert_ransac', 'epnp', 'normalized_DLT', 'epnp_gauss']
	parser.add_argument("--pnp", type=str, default="epnp_gauss", help="Enter your pnp algorithm name. ")
	
	# Parse the argument
	args = parser.parse_args()
	print("{} algorithm will be implemented. ".format(args.pnp))
	if args.onlyshow == 1:
		print('no pnp computation')
	return args

def load_extrinsicParams(args):

	# rotq_set = np.load("results/rotation_groundTruth.npy")  # true rotation
	# tvec_set = np.load("results/translation_groundTruth.npy") # true translation

	if args.onlyshow == 1: # Show the trajectory without running pnp
		if args.pnp == 'opencv_PnPRansac':
			rotq_set = np.load("results/rotation_estimated_opencvPnPRansac.npy")  # estimated rotation
			tvec_set = np.load("results/translation_estimated_opencvPnPRansac.npy") # estimated translation
		elif args.pnp == 'p3p_Grunert_ransac':
			rotq_set = np.load("results/rotation_estimated_Grunert1000.npy")  # estimated rotation
			tvec_set = np.load("results/translation_estimated_Grunert1000.npy") # estimated translation
		elif args.pnp == 'epnp':
			rotq_set = np.load("results/rotation_estimated_epnp.npy")  # estimated rotation
			tvec_set = np.load("results/translation_estimated_epnp.npy") # estimated translation
		elif args.pnp == 'normalized_DLT':
			rotq_set = np.load("results/rotation_estimated_DLT.npy")  # estimated rotation
			tvec_set = np.load("results/translation_estimated_DLT.npy") # estimated translation
		elif args.pnp == 'epnp_gauss':
			rotq_set = np.load("results/rotation_estimated_epnp_gauss.npy")  # estimated rotation
			tvec_set = np.load("results/translation_estimated_epnp_gauss.npy") # estimated translation
		else:
			rotq_set = np.load("results/rotation_estimated.npy")  # estimated rotation
			tvec_set = np.load("results/translation_estimated.npy") # estimated translation
	else: # pnp results
		rotq_set = np.load("results/rotation_estimated.npy")  # estimated rotation
		tvec_set = np.load("results/translation_estimated.npy") # estimated translation

	return rotq_set, tvec_set

def load_point_cloud():
	points3D_df = pd.read_pickle("data/points3D.pkl")
	xyz = np.vstack(points3D_df['XYZ'])
	rgb = np.vstack(points3D_df['RGB'])/255

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz)
	pcd.colors = o3d.utility.Vector3dVector(rgb)
	return pcd

def load_camera_pose(r, t):
	line_scale_wh = 0.09
	line_scale_z = 0.15
	line_set = o3d.geometry.LineSet()
	line_set.points = o3d.utility.Vector3dVector([[0, 0, 0], [line_scale_wh, line_scale_wh, line_scale_z], 
			                                            	[-line_scale_wh, line_scale_wh, line_scale_z], 
			                                            	[-line_scale_wh, -line_scale_wh, line_scale_z], 
			                                            	[line_scale_wh, -line_scale_wh, line_scale_z]])
	line_set.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
	line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], 
												  [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])

	line_set.rotate(R.from_quat(r).as_matrix())
	line_set.translate(t)
	return line_set

def get_transform_mtx(euler_rotation, translation, scale):
	r_mtx = R.from_euler('xyz', euler_rotation, degrees=True).as_matrix()
	transform_mtx = np.concatenate([scale * np.eye(3) @ r_mtx, translation.reshape(3, 1)], axis=1)
	transform_mtx = np.concatenate([transform_mtx, np.zeros([1, 4])], 0)
	transform_mtx[-1, -1] = 1.
	return transform_mtx

def main():
	
	args = parse_args() # Parse the argument

	if args.onlyshow == 1: # show the trajectory without running pnp
		rotq_set, tvec_set = load_extrinsicParams(args)
	else: 
		try:
			camPose = cameraPose(args.pnp)
		except:
			camPose = cameraPose('epnp_gauss')
			print('You enter the wrong name.\ndefault algorithm: EPnP + Gauss-Newton Optimization')

		data1 = data_loader()
		rtc = timeRecording_()

		# Start to run pnp
		imgIdx_start = 163 # validation image index = 163 ~ 292
		for i in range(imgIdx_start, len(data1.image_id_list)):
			data1.load_each_img_info(data1.image_id_list[i], 'i')

			# Find correspondance and solve pnp
			rotq, tvec = camPose.camera_pose_estimation((data1.kp_query, data1.desc_query),(data1.kp_train, data1.desc_train), data1.cameraIntrinsicParams, data1.distortionCoeffs)
			camPose.temporarily_store_poses(rotq, tvec, data1.rotq_gt, data1.tvec_gt)

		rtc.record()
		camPose.save_poses()

		rotq_set, tvec_set = load_extrinsicParams(args)

	# End of running pnp
	#
	# Start to plot the trajectory
	#
	vis = o3d.visualization.VisualizerWithKeyCallback()
	vis.create_window()

	## load point cloud
	pcd = load_point_cloud()
	vis.add_geometry(pcd)

	## load axes
	for i in range(len(rotq_set)):
	    pose_axes = load_camera_pose(rotq_set[i], tvec_set[i])
	    vis.add_geometry(pose_axes)


	# just set a proper initial camera view
	vc = vis.get_view_control()
	vc_cam = vc.convert_to_pinhole_camera_parameters()
	initial_cam = get_transform_mtx(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
	setattr(vc_cam, 'extrinsic', initial_cam)
	vc.convert_from_pinhole_camera_parameters(vc_cam)

	vis.run()
	vis.destroy_window()

if __name__ == '__main__':
	main()

	