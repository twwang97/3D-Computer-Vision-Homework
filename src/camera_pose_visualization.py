############################
#                          #
# Camera Calibration       #
#                          #
# After calibration, try   #
# to visualize the         #
# reletive poses of your   #
# cameras                  #
#                          #
# Author: David Wang       #
# Created on Nov. 14, 2022 #
#                          #
# modified from TA's code  #
# in 3DCV class (Fall2022) #
#                          #
############################

import numpy as np
import cv2 as cv
import open3d as o3d

class camera_pose_visualization:
	def __init__(self, camera_params, img_dim):
		self.inner_w = img_dim[0]
		self.inner_h = img_dim[1]
		self.K = camera_params['K']
		self.distCoeffs = camera_params['dist']
		self.rvecs = camera_params['rvecs']
		self.tvecs = camera_params['tvecs']
		self.imgs = camera_params['imgs']
		self.imgpoints = camera_params['imgpoints']

		self.projective_line_colors = [[1, 0, 0], [1, 0.5, 0], [0.75, 0.75, 0], [0, 1, 0], [0, 0.75, 0.75], [0, 0, 1], [1, 0, 1]]
		self.cam_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]
		self.cam_colors = [[1, 0, 0]] * len(self.cam_lines)

		coord = np.eye(4)
		coord[1, 1] = -1
		coord[2, 2] = -1
		self.coord = coord

	def expand_batch(self, _m, _batch_size):
	    b_m = np.repeat(np.expand_dims(_m, 0), _batch_size, axis=0)
	    return b_m

	def create_camera(self, _img, _r_mtx, _t_vec, _K):
		#
		# Inverse of transformation matrix
		#
		#      [ R | t ]
		#  T =  -------
		#      [ 0 | 1 ] 
		#           [ Ri | ti ]
		#  inv(T) =  ---------
		#           [ 0  | 1  ] 
		#
		# R = transpose(Ri) ; t =  - transpose(Ri) ti
		_r_mtx = np.transpose(_r_mtx)
		_t_vec = - _r_mtx @ _t_vec

		_h, _w = _img.shape[:2]
		verts = np.zeros((5, 3)).astype(np.float32)
		verts[1:3, 0] = _w
		verts[2:4, 1] = _h
		verts[:, 2] = 1.
		verts[4, 0] = _w / 2
		verts[4, 1] = _h / 2  
		verts = (self.expand_batch(np.linalg.inv(_K), 5) @ np.expand_dims(verts, -1))
		verts[4, -1] = 0.
		verts = (self.expand_batch(_r_mtx, 5) @ verts) + _t_vec

		viewing_frustum = o3d.geometry.LineSet()
		viewing_frustum.points = o3d.utility.Vector3dVector(verts.squeeze(-1))
		viewing_frustum.lines = o3d.utility.Vector2iVector(self.cam_lines)
		viewing_frustum.colors = o3d.utility.Vector3dVector(self.cam_colors)

		extrinsic = np.concatenate([_r_mtx, _t_vec], axis=-1)
		extrinsic = np.concatenate([extrinsic, np.zeros([1, 4], np.float32)], axis=0)
		extrinsic[-1, -1] = 1.

		img_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
		        o3d.geometry.Image(_img[..., ::-1].astype(np.uint8)),
		        o3d.geometry.Image(np.ones((_h, _w), np.float32)),
		        1.0, 2.0, False
		        )
		img_ = o3d.geometry.PointCloud.create_from_rgbd_image(
		        img_rgbd,
		        o3d.camera.PinholeCameraIntrinsic(_w, _h, _K[0, 0], _K[1, 1], _K[0, 2], _K[1, 2])
		        )
		img_.transform(extrinsic.astype(np.float64))

		return img_, viewing_frustum

	def render_line_colors(self):

		
		line_colors = []
		for i in range(self.inner_w):
		    for j in range(self.inner_h):
		        k = i % len(self.projective_line_colors)
		        line_colors.append(self.projective_line_colors[k])
		self.line_colors = line_colors

	def generate_lines_for_calibration(self, corners):
		num_points = corners.shape[0]
		line = o3d.geometry.LineSet()
		line.points = o3d.utility.Vector3dVector(corners)
		line.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(num_points - 1)])
		line.colors = o3d.utility.Vector3dVector(self.line_colors[:-1])
		return line

	def generate_chessboard_mesh(self):
		meshes = []
		for i in range(self.inner_h+1):
		    for j in range(self.inner_w+1):
		        color = [0, 0, 0] if (i + j) % 2 == 0 else [0.9, 0.9, 0.9]
		        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.2)
		        mesh.translate((i-1, - j, -0.2))
		        mesh.paint_uniform_color(color)
		        meshes.append(mesh)
		return meshes

	def convert_rt_form(self, rvec, tvec):
		r_mtx = cv.Rodrigues(rvec)[0]
		Rt = np.concatenate([r_mtx, tvec], -1)
		Rt = np.concatenate([Rt, np.zeros((1, 4))], 0)
		Rt[-1, -1] = 1.
		Rt = Rt @ self.coord
		r_mtx = Rt[:3, :3]
		tvec = Rt[:3, -1]
		tvec = np.expand_dims(tvec, -1)
		return r_mtx, tvec

	def estimate_chessboard_corners(self, r_mtx, tvec, imgpoint):
		imgpoint_ = np.concatenate([imgpoint, np.ones([imgpoint.shape[0], 1, 1], np.float32)], -1)
		campoint = self.batch_K_inv @ imgpoint_.transpose(0, 2, 1)
		worldpoint = self.expand_batch(r_mtx.transpose(), self.num_points) @ campoint
		_Rt = r_mtx.transpose() @ tvec
		s = worldpoint[:, -1] / _Rt[-1]
		s = np.expand_dims(np.repeat(s, 3, axis=1), -1)

		corners = worldpoint / s - _Rt
		corners = corners.reshape(corners.shape[0], -1)

		return corners

	def init(self):

		self.num_points = self.inner_w * self.inner_h
		self.batch_K_inv = self.expand_batch(np.linalg.inv(self.K), self.num_points)
		self.render_line_colors()

	def show_camera_relative_poses(self):
		
		# Initialize Open3D window
		vis = o3d.visualization.Visualizer()
		vis.create_window()

		# Chessboard
		meshes = self.generate_chessboard_mesh()
		for mesh in meshes:
			vis.add_geometry(mesh)

		self.init()
		
		for i, (img, rvec, tvec, imgpoint) in enumerate(zip(self.imgs, self.rvecs, self.tvecs, self.imgpoints)):

			r_mtx, tvec = self.convert_rt_form(rvec, tvec)
			chessboard_corners = self.estimate_chessboard_corners(r_mtx, tvec, imgpoint)
			calibration_lines = self.generate_lines_for_calibration(chessboard_corners)
			img_on_frustum, viewing_frustum = self.create_camera(img, r_mtx, tvec, self.K)

			vis.add_geometry(calibration_lines) # calibration_lines on chessboard
			vis.add_geometry(viewing_frustum) # frustum
			vis.add_geometry(img_on_frustum) # projective chessboard on frustum

		vis.run()