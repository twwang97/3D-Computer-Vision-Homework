############################
#                          #
# Author: David Wang       #
# Created on Oct. 20, 2022 #
#                          #
############################

import pandas as pd
import numpy as np
import cv2
import os

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID", "XYZ", "RGB", "DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

class data_loader:
	def __init__(self):

		self.cameraIntrinsicParams = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])    
		self.distortionCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])
		self.surfaceRGBcolor = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0]] # 6 surfaces on the cube
	  	
	  	# ground truth
		## images_df Table: 293 * 9
		# column: ['IMAGE_ID', 'NAME', 'TX', 'TY', 'TZ', 'QW', 'QX', 'QY', 'QZ']
		# (TX, TY, TZ): camera position 
		# (QW, QX, QY, QZ): rotation 
		self.images_df = pd.read_pickle("data/images.pkl")


		# TRAIN
		## train_df Table: 682468 * 6
		# column: ['POINT_ID', 'XYZ', 'RGB', 'IMAGE_ID', 'XY', 'DESCRIPTORS']
		# XYZ: point position
		# RGB: 1 * 3 rgb vector
		# XY: position of the descriptor in IMAGE_ID
		# DESCRIPTORS: 128-dimensional descriptors
		self.train_df = pd.read_pickle("data/train.pkl")

		# TRAIN
		## points3D_df Table: 112049 * 3
		# column: ['POINT_ID', 'XYZ', 'RGB']
		# XYZ: Point position
		self.points3D_df = pd.read_pickle("data/points3D.pkl")

		# QUERY
		## point_desc_df Table: 1234458 * 4 
		# column: ['POINT_ID', 'IMAGE_ID', 'XY', 'DESCRIPTORS']
		# If Point_ID is -1, then its 3D position is not available.
		# XY: position of keypoints
		# DESCRIPTORS: corresponding 128-dimensional descriptors
		self.point_desc_df = pd.read_pickle("data/point_desc.pkl")

		# Process train descriptors
		desc_df = average_desc(self.train_df, self.points3D_df)
		self.kp_train = np.array(desc_df["XYZ"].to_list())
		self.desc_train = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

		self.image_id_list = self.images_df["IMAGE_ID"].to_list() # ID range: 1 ~ 293

		img_directory = 'data/frames'
		self.img_name_list = [img_i_name for img_i_name in os.listdir(img_directory) if img_i_name.find('valid') != -1] # find validation images
		self.img_name_list.sort(key=lambda x: int(x[9:-4]))
		self.img_valid_path_list = []
		for img_i_name in self.img_name_list:
			filename = os.path.join(img_directory, img_i_name)
			self.img_valid_path_list.append(filename)

		print('Initial data are loaded!')

	def laod_camera_poses(self):
		self.rotq_set = np.load("results/rotation_estimated.npy") # estimated rotation
		self.tvec_set = np.load("results/translation_estimated.npy") # estimated translation
		# self.rotq_set = np.load("results/rotation_groundTruth.npy") # true rotation
		# self.tvec_set = np.load("results/translation_groundTruth.npy") # true translation

	def get_query_image(self, img_idx, img_col):
		# self.img_i_name = ((self.images_df.loc[self.images_df["IMAGE_ID"] == img_idx])["NAME"].values)[0]
		# self.img_i = cv2.imread("data/frames/" + self.img_i_name, cv2.IMREAD_GRAYSCALE)
		if img_col == 'n': 
			self.img_i = cv2.imread(self.img_valid_path_list[img_idx])
			self.img_shape = (self.img_i.shape[1], self.img_i.shape[0])
		
	def get_query_keypoint_descriptor(self, img_idx):
		points = self.point_desc_df.loc[self.point_desc_df["IMAGE_ID"]==img_idx]
		self.kp_query = np.array(points["XY"].to_list()) # query keypoints
		self.desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32) # query descriptors

	def get_camera_pose_groudtruth_id(self, img_idx):
		ground_truth = self.images_df.loc[self.images_df["IMAGE_ID"]==img_idx]
		self.rotq_gt = ground_truth[["QX", "QY", "QZ", "QW"]].values
		self.tvec_gt = ground_truth[["TX", "TY", "TZ"]].values

	def get_camera_pose_groudtruth_name(self, img_idx):
		ground_truth = self.images_df.loc[self.images_df["NAME"]==self.img_name_list[img_idx]]
		self.rotq_gt = ground_truth[["QX", "QY", "QZ", "QW"]].values
		self.tvec_gt = ground_truth[["TX", "TY", "TZ"]].values

	def load_each_img_info(self, img_idx, img_col):
		# Load the query image
		self.get_query_image(img_idx, img_col)

		# Load the query keypoints and descriptors
		self.get_query_keypoint_descriptor(img_idx)

		# Get camera pose groudtruth 
		if img_col == 'i': # ID
			self.get_camera_pose_groudtruth_id(img_idx)
			print("\nprocessing {}/{}".format(img_idx, len(self.image_id_list)))
		elif img_col == 'n': # name
			self.get_camera_pose_groudtruth_name(img_idx)

		# print("\n{}/{}, {}".format(img_idx, len(self.image_id_list), self.img_i_name))
		