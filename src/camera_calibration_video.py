############################
#                          #
# Camera Calibration       #
# from a video source      #
#                          #
# Author: David Wang       #
# Created on Nov. 14, 2022 #
#                          #
############################

# Geometric camera calibration estimates the 
# parameters of a lens and image sensor of an image.

import numpy as np
import cv2 as cv

class Calibrator:
	def __init__(self, args):
	    self.args = args
	    self.inner_w = args.w
	    self.inner_h = args.h

	    # root mean square (RMS) re-projection error
	    # RMS in the range of [0, 1] pixel in good calibration
	    self.calibrationRMSE = 10 

	    # termination criteria
	    self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	    self.objp = np.zeros((self.inner_w * self.inner_h, 3), np.float32)
	    self.objp[:,:2] = np.mgrid[0:self.inner_h, 0:self.inner_w].T.reshape(-1,2)
	    # Arrays to store object points and image points from all the images.
	    self.objpoints = [] # 3d point in real world space
	    self.imgpoints = [] # 2d points in image plane.
	    self.imgs = []
	    self.final_camera_params, self.camera_params_set = None, None

	def run_video(self):
	    self.video_reader = cv.VideoCapture(self.args.input)
	    self.load_images()
	    assert len(self.imgs) >= 4, print('=> Error: need a least 4 images to calibrate')

	    self.calibrationRMSE, self.K, self.dist, self.rvecs, self.tvecs = self.calibrate()
	    self.camera_params_set = {'K': self.K, 'dist': self.dist, 'rvecs': self.rvecs, 'tvecs': self.tvecs, 'imgs': self.imgs, 'imgpoints': self.imgpoints}
	    self.save_result()

	def load_images(self):

	    #for fname in images:
	    print('=> press <space> to add an image')
	    print('=> press <q> to quit and exit')
	    while True:
	        #img = cv.imread(fname)
	        ret, img = self.video_reader.read()
	        if not ret: break
	        cv.imshow('calibration video', img)
	        key = cv.waitKey(10) #& 0xFF
	        if key == ord(' '): # space key
	            self.imgs.append(img)
	            print('Saving the image {} / 4'.format(len(self.imgs)))
	        if key == ord('q'): # key 'q'
	            break
	    cv.destroyAllWindows()

	def calibrate(self):

	    print('Start the calibration process')

	    for img in self.imgs:
	        # Find the chess board corners
	        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	        ret, corners = cv.findChessboardCorners(gray, (self.inner_h, self.inner_w), None)
	        # If found, add object points, image points (after refining them)
	        if ret == True:
	            self.objpoints.append(self.objp)
	            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), self.criteria)
	            self.imgpoints.append(corners2)
	            # Draw and display the corners
	            img_ = cv.UMat(img)
	            cv.drawChessboardCorners(img_, (self.inner_h, self.inner_w), corners2, ret)
	            cv.imshow('img', img_)
	            cv.waitKey(45)

	    cv.destroyAllWindows()
	    
	    assert len(self.objpoints) > 0, print('=> Error: no corner detected')
	    assert len(self.objpoints) >= 4, print('=> Error: need a least 4 images to calibrate')
	    print('Corners are found in {} images'.format(len(self.objpoints)))

	    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	    return cv.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

	def save_result(self):
		np.set_printoptions(precision=3, suppress=True)
		print('Camera Intrinsic Matrix')
		print(self.K)
		print('Distortion Coefficients')
		print(self.dist)
		print('Overall RMS re-projection error: {}'.format(self.calibrationRMSE))

		self.final_camera_params = {'K': self.K, 'dist': self.dist, 'rms': self.calibrationRMSE}
		np.save(self.args.output, self.final_camera_params)
