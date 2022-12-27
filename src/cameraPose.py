############################
#                          #
# Camera Pose Estimation   #
#                          #
# 2D-3D correspondence     #
# and PnP Computation      #
#                          #
# Author: David Wang       #
# Created on Oct. 20, 2022 #
#                          #
############################

from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2

from src.p3p_Grunert import *
from src.EPnP import EPnP
from src.DLT import DLT

class cameraPose:
	def __init__(self, solver=1, minInliers=2000):
		if len(solver) == 16: # solver == 'opencv_PnPRansac'
			self.solver = 0 # opencv solvePnPRansac
		elif len(solver) == 18: # solver == 'p3p_Grunert_ransac'
			self.solver = 1 # RANSAC + P3P (Grunert method)
		elif len(solver) == 4: # solver == 'epnp'
			self.solver = 2 # Efficient PnP (EPnP)
			self.epnp = EPnP()
		elif len(solver) == 14: # solver == 'normalized_DLT'
			self.solver = 3 # normalized Direct Linear Transform (DLT) 
			self.dlt = DLT()
		elif len(solver) == 10: # solver == 'epnp_gauss'
			self.solver = 4 # EPnP + Gauss-Newton Optimization
			self.epnp = EPnP()

		self.rotq_estimated_set = None
		self.tvec_estimated_set = None
		self.rotq_gt_set = None
		self.tvec_gt_set = None
		self.img_shape = [1920, 1080, 3]
		self.minInliers = minInliers
		self.inliers_errorThreshold = 0.5
		self.minOfRANSACiterations = 20
		self.maxOfRANSACiterations = 200 # 1000
		
	def correspondence_estimation(self, query_set, train_set):
		keypoints_1, descriptors_1 = query_set
		keypoints_2, descriptors_2 = train_set

		bf = cv2.BFMatcher()
		matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

		gmatches = []
		for m,n in matches:
		    if m.distance < 0.75 * n.distance:
		        gmatches.append(m)

		imagePoints = np.empty((0, 2)) # 2D vector
		objectPoints = np.empty((0, 3)) # 3D vector
		for match in gmatches:
		    imagePoints = np.vstack((imagePoints, keypoints_1[match.queryIdx]))
		    objectPoints = np.vstack((objectPoints, keypoints_2[match.trainIdx]))

		return imagePoints, objectPoints

	def opencv_pnp_ransac(self, query_set, train_set, imagePoints, objectPoints, cameraIntrinsicParams=0, distortionCoeffs=0):
		return cv2.solvePnPRansac(objectPoints, imagePoints, cameraIntrinsicParams, distortionCoeffs)

	def undistortImgPts(self, point2D, cameraIntrinsicParams, distortionCoeffs, NumOfIterations=3):
		# Undistort the image points
		# reference
		# https://yangyushi.github.io/code/2020/03/04/opencv-undistort.html

		k1, k2, p1, p2, k3 = distortionCoeffs
		fx, fy = cameraIntrinsicParams[0, 0], cameraIntrinsicParams[1, 1]
		cx, cy = cameraIntrinsicParams[:2, 2]
		xy_corrected = np.zeros((point2D.shape[0], point2D.shape[1]))
		for i in range(point2D.shape[0]):
			x, y = point2D[i, :]
			x = (x - cx) / fx
			x0 = x
			y = (y - cy) / fy
			y0 = y
			for _ in range(NumOfIterations):
			    r2 = x ** 2 + y ** 2
			    k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
			    delta_x = 2 * p1 * x*y + p2 * (r2 + 2 * x**2)
			    delta_y = p1 * (r2 + 2 * y**2) + 2 * p2 * x*y
			    x = (x0 - delta_x) * k_inv
			    y = (y0 - delta_y) * k_inv
			xy_corrected[i, :] = [x * fx + cx, y * fy + cy]
		return xy_corrected

	
	def p3p_Grunert(self, query_set, train_set, imagePoints, objectPoints, cameraIntrinsicParams=0, distortionCoeffs=0):

		
		is_valid_solution = False
		numOfPoints = 3 # n for pnp algorithm
		while not is_valid_solution:

			# Get lengths
			is_colinear_3pts = True
			while is_colinear_3pts:
				
				# Generate a uniform random sample from np.arange(shape[0]) of size (3+numValid):
				randomIdx = np.random.choice(objectPoints.shape[0], numOfPoints)
				# testIdx = np.arange(objectPoints.shape[0]) # 0 ~ max index
				
				
				X, U = objectPoints[randomIdx], imagePoints[randomIdx]

				if np.linalg.norm(np.cross(X[1]-X[0], X[2]-X[0])) < 1e-5:
					print("\tpnpRansac: 3 Points are colinear!")
					is_colinear_3pts = True
				else:
					is_colinear_3pts = False

			v = np.dot(np.linalg.inv(cameraIntrinsicParams), np.hstack((U, np.ones((U.shape[0], 1)))).T).T
			innerProd_set = cosine_angle(v[0], v[1]), cosine_angle(v[0], v[2]), cosine_angle(v[1], v[2])
			distance_set = np.linalg.norm(X[0] - X[1]), np.linalg.norm(X[0] - X[2]), np.linalg.norm(X[1] - X[2])
			lengths = findLengths(innerProd_set, distance_set)

			try:
				# find all sets of rotation and translation
				rot_mtx_set, t_vec_set, lambda_set = [], [], []
				for i in range(len(lengths)):
					trans_trilateration_set = trilateration(lengths[i], X)
					for trans_vec in trans_trilateration_set:

						lambda_i = np.linalg.norm(X[0] - trans_vec) / np.linalg.norm(v[0])
						if lambda_i < 0:
						    continue

						rot_mtx = rot2vec(X[0] - trans_vec, lambda_i * v[0])
						# rot_1 = R.from_rotvec(X[0] - trans_vec)
						# angle = R.from_rotvec(lambda_i*v[0]) * rot_1.inv()
						# rot_mtx = angle.as_matrix()
						

						if np.linalg.det(rot_mtx) - 1 > 1e-5:
						    continue
						# elif np.linalg.det(rot_mtx) + 1 < -1e-5:
						#    continue

						rot_mtx_set.append(rot_mtx)
						t_vec_set.append(trans_vec)
						lambda_set.append(lambda_i)

				# Select the best set of rotation and translation
				# from all correspondences
				rot_mtx_i, trans_i, lambda_i, inliersIdx, error_min_i = 0, 0, 0, [], float('inf')
				V = np.dot(np.linalg.inv(cameraIntrinsicParams), np.hstack((U, np.ones((U.shape[0], 1)))).T).T
				for i in range(len(rot_mtx_set)):
				    err = 0
				    for j in range(len(X)):
				        err += np.linalg.norm(rot_mtx_set[i].dot(X[j] - t_vec_set[i]) - lambda_set[i] * V[j])
				    err /= len(X)
				    if err < error_min_i:
				        error_min_i = err
				        rot_mtx_i = rot_mtx_set[i]
				        trans_i = t_vec_set[i]
				        lambda_i = lambda_set[i]

				R_opt = R.from_matrix(rot_mtx_i)
				rotq_i = R_opt.as_quat()
				# print("Best: T = {}\nR = {}".format(trans_i, rot_mtx_i))

				is_valid_solution = True

			except:
				print("\tAn exception occurred")
				is_valid_solution = False

		# find the inliers
		for i in range(imagePoints.shape[0]):
			# print(rot_mtx_i.shape, X.shape, trans_i.shape, lambda_i, V.shape)
			Vj = np.dot(np.linalg.inv(cameraIntrinsicParams), np.hstack((imagePoints, np.ones((imagePoints.shape[0], 1)))).T).T
			err_i = np.linalg.norm(rot_mtx_i.dot(objectPoints[i] - trans_i) - lambda_i * Vj[i])
			if err_i < self.inliers_errorThreshold:
				inliersIdx.append(i)

		return error_min_i, rotq_i, trans_i, inliersIdx

	def p3p_Grunert_ransac(self, query_set, train_set, imagePoints, objectPoints, cameraIntrinsicParams=0, distortionCoeffs=0):

		# imagePoints_corrected = cv2.undistortPoints(imagePoints, cameraIntrinsicParams, distortionCoeffs)
		# imagePoints_corrected = imagePoints_corrected.reshape(imagePoints_corrected.shape[0], 2)
		imagePoints_corrected = self.undistortImgPts(imagePoints, cameraIntrinsicParams, np.append(distortionCoeffs, [0]), NumOfIterations=3)

		rotq_final, trans_final, error_ransac_min = 0, 0, float('inf')
		count_ransac_iter = 0
		is_convergent_solution = False
		while not is_convergent_solution:
			count_ransac_iter += 1
			err_iter, rotq, tvec, inliers = self.p3p_Grunert(query_set, train_set, imagePoints_corrected, objectPoints, cameraIntrinsicParams, distortionCoeffs)
			if err_iter < error_ransac_min:
				error_ransac_min = err_iter
				rotq_final = rotq
				trans_final = tvec

			if len(inliers) > self.minInliers and count_ransac_iter > self.minOfRANSACiterations:
				is_convergent_solution = True
			elif count_ransac_iter > self.maxOfRANSACiterations:
				is_convergent_solution = True
			elif count_ransac_iter % 20 == 1:
				print("\trunning ransac {}/{} ...".format(count_ransac_iter, self.maxOfRANSACiterations))
		print("error = {:.4f}, inliers count = {}".format(error_ransac_min, len(inliers)))
		# print("Best: T = {}\nR = {}".format(trans_final, rotq_final))

		return rotq_final, trans_final

	def efficient_pnp(self, query_set, train_set, imagePoints, objectPoints, cameraIntrinsicParams=0, distortionCoeffs=0):
		imagePoints = self.undistortImgPts(imagePoints, cameraIntrinsicParams, np.append(distortionCoeffs, [0]), NumOfIterations=3)
		A = np.zeros((3, 4))
		A[0:3, 0:3] = cameraIntrinsicParams

		error, Rt, Cc, Xc = \
			self.epnp.efficient_pnp(objectPoints.reshape(objectPoints.shape[0], objectPoints.shape[1], 1), \
										imagePoints.reshape(imagePoints.shape[0], imagePoints.shape[1], 1),  \
										A) 

		print("\terror = {:.4f}".format(error)) # error 

		r = R.from_matrix(Rt[0:3, 0:3])
		rotq = r.as_quat()
		tranc_vec = Rt[0:3, 3]
		return rotq, tranc_vec

	def efficient_pnp_Gauss(self, query_set, train_set, imagePoints, objectPoints, cameraIntrinsicParams=0, distortionCoeffs=0):
		imagePoints = self.undistortImgPts(imagePoints, cameraIntrinsicParams, np.append(distortionCoeffs, [0]), NumOfIterations=3)
		A = np.zeros((3, 4))
		A[0:3, 0:3] = cameraIntrinsicParams

		error, Rt, Cc, Xc = \
			self.epnp.efficient_pnp_gauss(objectPoints.reshape(objectPoints.shape[0], objectPoints.shape[1], 1), \
										imagePoints.reshape(imagePoints.shape[0], imagePoints.shape[1], 1),  \
										A) 

		print("\terror = {:.4f}".format(error)) # error 

		r = R.from_matrix(Rt[0:3, 0:3])
		rotq = r.as_quat()
		tranc_vec = Rt[0:3, 3]
		return rotq, tranc_vec

	def normalized_dlt(self, query_set, train_set, imagePoints, objectPoints, cameraIntrinsicParams=0, distortionCoeffs=0):
		imagePoints = self.undistortImgPts(imagePoints, cameraIntrinsicParams, np.append(distortionCoeffs, [0]), NumOfIterations=3)
		Rt, err = self.dlt.DLTcalib(objectPoints, imagePoints, cameraIntrinsicParams, 3)
		# print('Matrix')
		# print(Rt)
		print("\terror = {:.4f}".format(err)) # error
		r = R.from_matrix(Rt[0:3, 0:3])
		rotq = r.as_quat()
		tranc_vec = Rt[0:3, 3]
		return rotq, tranc_vec

	def camera_pose_estimation(self, query_set, train_set, cameraIntrinsicParams=0, distortionCoeffs=0):

		# Get 3D-2D correspondences
		imagePoints, objectPoints = self.correspondence_estimation(query_set, train_set)

		# Finds an object pose from 3D-2D correspondences using the RANSAC scheme
		if self.solver == 0: # solved by opencv toolkit
			retval, rvec, tvec, inliers = self.opencv_pnp_ransac(query_set, train_set, imagePoints, objectPoints, cameraIntrinsicParams, distortionCoeffs)
			rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # quaternion format: (x, y, z, w) 
			rotq, tvec = self.inverse_transformation_mtx(rotq.reshape(4), tvec.reshape(3)) # estimated rotation and translation
			return rotq, tvec
		elif self.solver == 1: # solved by RANSAC and P3P (Grunert method)
			return self.p3p_Grunert_ransac(query_set, train_set, imagePoints, objectPoints, cameraIntrinsicParams, distortionCoeffs)
		elif self.solver == 2: # solved by EPnP
			rotq, tvec = self.efficient_pnp(query_set, train_set, imagePoints, objectPoints, cameraIntrinsicParams, distortionCoeffs)
			rotq, tvec = self.inverse_transformation_mtx(rotq.reshape(4), tvec.reshape(3)) # estimated rotation and translation
			return rotq, tvec
		elif self.solver == 3: # solved by DLT
			rotq, tvec = self.normalized_dlt(query_set, train_set, imagePoints, objectPoints, cameraIntrinsicParams, distortionCoeffs)
			rotq, tvec = self.inverse_transformation_mtx(rotq.reshape(4), tvec.reshape(3)) # estimated rotation and translation
			return rotq, tvec
		elif self.solver == 4: # solved by EPnP and Gauss-Newton Optimization
			rotq, tvec = self.efficient_pnp_Gauss(query_set, train_set, imagePoints, objectPoints, cameraIntrinsicParams, distortionCoeffs)
			rotq, tvec = self.inverse_transformation_mtx(rotq.reshape(4), tvec.reshape(3)) # estimated rotation and translation
			return rotq, tvec

	def inverse_transformation_mtx(self, r_quat, t_vec):
		#      [ R | t ]
		#  T =  -------
		#      [ 0 | 1 ] 
		#           [ R' | t' ]
		#  inv(T) =  ---------
		#           [ 0  | 1  ] 
		# R = transpose(R') ; t =  - transpose(R') t'

		r_quat = np.array([-r_quat[0], -r_quat[1], -r_quat[2], r_quat[3]])
		r = R.from_quat(r_quat)
		r_mtx = r.as_matrix()
		t_vec = -r_mtx @ t_vec
		return r_quat, t_vec

	def temporarily_store_poses(self, rotq_est, tvec_est, rotq_gt, tvec_gt):

		# The pose of an image is represented as 
		# the projection from world to the camera coordinate system.
		rotq_gt, tvec_gt = self.inverse_transformation_mtx(rotq_gt.reshape(4), \
								   tvec_gt.reshape(3)) # true rotation and translation

		# store all rotation and translation
		if type(self.rotq_estimated_set) == type(None): # save data for the 1st time
			self.rotq_estimated_set = np.array(rotq_est)
			self.tvec_estimated_set = np.array(tvec_est)
			self.rotq_gt_set = np.array(rotq_gt)
			self.tvec_gt_set = np.array(tvec_gt)
		else:
			self.rotq_estimated_set = np.vstack((self.rotq_estimated_set, rotq_est)) 
			self.tvec_estimated_set = np.vstack((self.tvec_estimated_set, tvec_est)) 
			self.rotq_gt_set = np.vstack((self.rotq_gt_set, rotq_gt)) 
			self.tvec_gt_set = np.vstack((self.tvec_gt_set, tvec_gt)) 

	def save_poses(self):
		np.save("results/rotation_estimated", self.rotq_estimated_set)
		np.save("results/translation_estimated", self.tvec_estimated_set)
		np.save("results/rotation_groundTruth", self.rotq_gt_set)
		np.save("results/translation_groundTruth", self.tvec_gt_set)
