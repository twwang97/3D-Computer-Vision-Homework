############################
#                          #
# Camera Pose Estimation & #
# Augmented Reality (AR)   #
#                          #
# 2D-3D correspondence     #
# method: P3P + RANSAC     #
#                          #
# Author: David Wang       #
# Created on Oct. 20, 2022 #
#                          #
############################

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

from src.data_loader import data_loader
from src.cameraPose import cameraPose
from src.timeRecording import timeRecording_

def parse_args():
    print('2D-3D Correspondences will be matched and Augmented Reality (AR) will be presented. ')
    parser = argparse.ArgumentParser(description="Match 2D-3D Correspondences and Plot the Trajectory")
    parser.add_argument("--onlyshow", type=int, default=0, help="Show the cube without running pnp, then enter 1. Enter 2 then ground truth data will be presented")
    parser.add_argument("--videopath", type=str, default='results/cubeVideo.mp4', help="Enter your video path. ")
    # pnp option: ['opencv_PnPRansac', 'p3p_Grunert_ransac', 'epnp', 'normalized_DLT', 'epnp_gauss']
    parser.add_argument("--pnp", type=str, default="epnp_gauss", help="Enter your pnp algorithm name. ")
    
    # Parse the argument
    args = parser.parse_args()

    if args.onlyshow == 2:
        print('Only ground truth will be used!')
    else:
        print("{} algorithm will be implemented. ".format(args.pnp))
        if args.onlyshow == 1:
            print('no pnp computation')
    return args

def load_extrinsicParams(args, camPose=None):

    # rotq_set = np.load("results/rotation_groundTruth.npy")  # true rotation
    # tvec_set = np.load("results/translation_groundTruth.npy") # true translation

    if args.onlyshow == 1: # Show the cube without running pnp
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

        for i in range(len(rotq_set)):
            rot_quat = rotq_set[i]
            trans_vec = tvec_set[i]
            rotq_set[i], tvec_set[i] = camPose.inverse_transformation_mtx(rot_quat.reshape(4), trans_vec.reshape(3))

    elif args.onlyshow == 0: # pnp results
        rotq_set = np.load("results/rotation_estimated.npy")  # estimated rotation
        tvec_set = np.load("results/translation_estimated.npy") # estimated translation

    return rotq_set, tvec_set

def get_extrinsic_transform_mtx(rot_quat, trans_vec, scale):
    r_mtx = R.from_quat(rot_quat).as_matrix()
    transform_mtx = np.concatenate([scale * np.eye(3) @ r_mtx, trans_vec.reshape(3, 1)], axis=1)
    return transform_mtx

def cube_3d_vertices():
	vertices_3d = [[ 1.17,       -0.79,        1.23      ],
				 [ 1.40907619, -0.79,        1.15690707],
				 [ 1.24309293, -0.79,        1.46907619],
				 [ 1.48216912, -0.79,        1.39598326],
				 [ 1.17,       -0.54,        1.23      ],
				 [ 1.40907619, -0.54,        1.15690707],
				 [ 1.24309293, -0.54,        1.46907619],
				 [ 1.48216912, -0.54,        1.39598326]]
	return np.array(vertices_3d)

def get_cube_transformedPts(pts, v0, v1, v2, v3, intrinsicMtx, extrinsicMtx, surfaceID):
    numPoints = 8 # number of points per side of the cube
    head = []

    vec0 = v2 - v0
    for i in range(numPoints):
        head.append(v0 + vec0 * i / (numPoints-1))

    vec1 = v1 - v0
    for i in range(numPoints):
        for j in range(numPoints):
            point_3d = head[i] + vec1 * j  / (numPoints-1)
            point_2d = intrinsicMtx.dot(extrinsicMtx.dot(np.append(point_3d, 1.0)))
            point_2d /= point_2d[-1]
            point_2d = [int(point_2d[0]), int(point_2d[1])]
            pts.append((point_2d, point_3d[-1], surfaceID))

    return pts

def get_points_on_cube(vertices, cameraIntrinsicParams, extrinsicMatrix):

    pts = []
    pts = get_cube_transformedPts(pts, vertices[0], vertices[1], vertices[2], vertices[3], cameraIntrinsicParams, extrinsicMatrix, 0)
    pts = get_cube_transformedPts(pts, vertices[4], vertices[5], vertices[6], vertices[7], cameraIntrinsicParams, extrinsicMatrix, 1)
    pts = get_cube_transformedPts(pts, vertices[0], vertices[1], vertices[4], vertices[5], cameraIntrinsicParams, extrinsicMatrix, 2)
    pts = get_cube_transformedPts(pts, vertices[2], vertices[3], vertices[6], vertices[7], cameraIntrinsicParams, extrinsicMatrix, 3)
    pts = get_cube_transformedPts(pts, vertices[0], vertices[2], vertices[4], vertices[6], cameraIntrinsicParams, extrinsicMatrix, 4)
    pts = get_cube_transformedPts(pts, vertices[1], vertices[3], vertices[5], vertices[7], cameraIntrinsicParams, extrinsicMatrix, 5)
    pts.sort(key=lambda x: -x[1])
    return pts

def main():

    args = parse_args() # Parse the argument
    data1 = data_loader()

    if args.onlyshow == 1: # show the cube without running pnp
        camPose = cameraPose(args.pnp)
        rotq_set, tvec_set = load_extrinsicParams(args, camPose)
    elif args.onlyshow == 0: 
        try:
            camPose = cameraPose(args.pnp)
        except:
            camPose = cameraPose('epnp_gauss')
            print('You enter the wrong name.\ndefault algorithm: EPnP + Gauss-Newton Optimization')

        
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
    # Start to plot the cube
    #

    cube_vertices  = cube_3d_vertices()

    img_list = []
    for i in range(len(data1.img_valid_path_list)):
        
        data1.load_each_img_info(i, 'n')
        img = data1.img_i

        if args.onlyshow == 2:
            rot_quat = data1.rotq_gt # ground truth data
            trans_vec = data1.tvec_gt # ground truth data
        else:
            rot_quat = rotq_set[i]
            trans_vec = tvec_set[i]

        cameraExtrinsicParams = get_extrinsic_transform_mtx(rot_quat.reshape(4), trans_vec, 1)
        pts = get_points_on_cube(cube_vertices, data1.cameraIntrinsicParams, cameraExtrinsicParams)
        for (pt, depth, surfaceID) in pts:
            cv2.circle(img, tuple(pt), 2, data1.surfaceRGBcolor[surfaceID], 3)
        img_list.append(img)

    out = cv2.VideoWriter(args.videopath, cv2.VideoWriter_fourcc('m','p','4','v'), 15, data1.img_shape)
    for frame in img_list:
        out.write(frame)

    out.release()

if __name__ == '__main__':
    main()
