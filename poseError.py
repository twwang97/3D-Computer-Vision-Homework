############################
#                          #
# Pose Error Estimation    #
#                          #
# Author: David Wang       #
# Created on Oct. 20, 2022 #
#                          #
############################

from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse

    
def main():

	print('Start to estimate the pose error. ')

	rotq_gt_set = np.load("results/rotation_groundTruth.npy")  # true rotation
	tvec_gt_set = np.load("results/translation_groundTruth.npy") # true translation

	rotq_estimated_set = np.load("results/rotation_estimated.npy")  # estimated rotation
	tvec_estimated_set = np.load("results/translation_estimated.npy") # estimated translation

	# calculate the error
	tvec_diff = np.zeros((rotq_estimated_set.shape[0], 3))
	angle_diff = np.zeros((rotq_estimated_set.shape[0], 3))
	for i in range(rotq_estimated_set.shape[0]):
		rotq_estimated = rotq_estimated_set[i, :]
		tvec_estimated = tvec_estimated_set[i, :]
		rotq_gt = rotq_gt_set[i, :]
		tvec_gt = tvec_gt_set[i, :]

		## Pose error: rotation
		rotq_estimated = R.from_quat(rotq_estimated)
		angle = R.from_quat(rotq_gt) * rotq_estimated.inv()
		angle_diff[i, :] = angle.as_rotvec()

		## Pose error: translation
		tvec_diff[i, :] = tvec_estimated - tvec_gt
		
	error_angle = np.linalg.norm(angle_diff, axis=1)
	error_tvec = np.linalg.norm(tvec_diff, axis=1)
	print("Median Rotation Error = {:.6f}".format(np.median(error_angle)))
	print("Median Translation Error = {:.6f}".format(np.median(error_tvec)))

if __name__ == '__main__':
	main()