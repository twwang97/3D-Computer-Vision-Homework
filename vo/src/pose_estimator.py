############################
#                          #
# Pose Estimation          #
# from Epipolar Geometry   #
#                          #
# Author: David Wang       #
# Created on Nov. 14, 2022 #
#                          #
# mainly modified from     #
# laavanyebahl's github    #
#                          #
############################

# modified from
# https://github.com/laavanyebahl/3D-Reconstruction-and-Epipolar-Geometry

import numpy as np
import math
import cv2
import random

from vo.src.utils.utils_pose import refineF

'''
# Eight Point Algorithm
    Input:  normalized pts1, Nx2 Matrix
            normalized pts2, Nx2 Matrix
    Output: E, the essential matrix
'''
def computeEssentialMtx_8pts(pts1_scaled, pts2_scaled):

    A_f = np.zeros((pts1_scaled.shape[0], 9))

    for i in range(pts1_scaled.shape[0]):
        A_f[i, :] = [ pts2_scaled[i,0]*pts1_scaled[i,0] , 
                    pts2_scaled[i,0]*pts1_scaled[i,1] , 
                    pts2_scaled[i,0], 
                    pts2_scaled[i,1]*pts1_scaled[i,0] , 
                    pts2_scaled[i,1]*pts1_scaled[i,1] , 
                    pts2_scaled[i,1], 
                    pts1_scaled[i,0], 
                    pts1_scaled[i,1], 
                    1 ]

    # print('A shape: ',A_f.shape)

    u, s, vh = np.linalg.svd(A_f)
    v = vh.T
    E = v[:, -1].reshape(3,3)

    E = refineF(E, pts1_scaled, pts2_scaled)
    # print('refined f :', f)
    # print('rank of refined f :', np.linalg.matrix_rank(f))

    return E


'''
# Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation

    # TRIANGULATION
    # http://cmp.felk.cvut.cz/cmp/courses/TDV/2012W/lectures/tdv-2012-07-anot.pdf

    # Form of Triangulation :
    #
    # x = C.X
    #
    # |x|             | u |
    # |y| =   C(3x4). | v |
    # |1|             | w |
    #                 | 1 |
    #
    # 1 = C_3 . X
    #
    # x_i . (C_3_i.X_i) = C_1_i.X_i
    # y_i.  (C_3_i.X_i) = C_2_i.X_i

    # Subtract RHS from LHS and equate to 0
    # Take X common to get AX=0
    # Solve for X with SVD
    # for 2 points we have four equation

    P_i = []

    for i in range(pts1.shape[0]):
        A = np.array([   pts1[i,0]*C1[2,:] - C1[0,:] ,
                         pts1[i,1]*C1[2,:] - C1[1,:] ,
                         pts2[i,0]*C2[2,:] - C2[0,:] ,
                         pts2[i,1]*C2[2,:] - C2[1,:]   ])

        # print('A shape: ', A.shape)
        u, s, vh = np.linalg.svd(A)
        v = vh.T
        X = v[:,-1]
        # NORMALIZING
        X = X/X[-1]
        # print(X)
        P_i.append(X)

    P_i = np.asarray(P_i)

    # print('P_i: ', P_i)

    # MULTIPLYING TOGETHER WIH ALL ELEMENET OF Ps
    pts1_out = np.matmul(C1, P_i.T )
    pts2_out = np.matmul(C2, P_i.T )

    pts1_out = pts1_out.T
    pts2_out = pts2_out.T

    # NORMALIZING
    for i in range(pts1_out.shape[0]):
        pts1_out[i,:] = pts1_out[i,:] / pts1_out[i, -1]
        pts2_out[i,:] = pts2_out[i,:] / pts2_out[i, -1]

    # NON - HOMOGENIZING
    pts1_out = pts1_out[:, :-1]
    pts2_out = pts2_out[:, :-1]

    # print('pts2_out shape: ', pts2_out.shape)
    # print('pts1_out: ', pts1_out)
    # print('pts2_out: ', pts2_out)

    # CALCULATING REPROJECTION ERROR
    reprojection_err = 0
    for i in range(pts1_out.shape[0]):
        reprojection_err = reprojection_err  + np.linalg.norm( pts1[i,:] - pts1_out[i,:] )**2 + np.linalg.norm( pts2[i,:] - pts2_out[i,:] )**2
    # print(reprojection_err)

    # NON-HOMOGENIZING
    P_i = P_i[:, :-1]

    return P_i, reprojection_err

def candidate_poses(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    return M2s

def ransacE(pts1, pts2):
    # E = computeEssentialMtx_8pts(pts1, pts2)

    max_inliers  =  -np.inf
    inliers_best = np.zeros(pts1.shape[0], dtype=bool)
    points_index_best  = None
    threshold = 1e-3

    epochs = 20
    for e in range(epochs):
        points_index = random.sample(range(0, pts1.shape[0]), 8)
        # print(points_index)
        eightpoints_1 = []
        eightpoints_2 = []
        for point in points_index:
            eightpoints_1.append(pts1[point, :])
            eightpoints_2.append(pts2[point, :])
        eightpoints_1 = np.asarray(eightpoints_1)
        eightpoints_2 = np.asarray(eightpoints_2)

        E_e = computeEssentialMtx_8pts(eightpoints_1, eightpoints_2)
        num_inliers = 0
        inliers = np.zeros(pts1.shape[0], dtype=bool)
        for k in range(pts1.shape[0]):
            X2 = np.asarray(  [pts2[k,0], pts2[k,1], 1] )
            X1 = np.asarray(  [pts1[k,0], pts1[k,1], 1] )

            if abs(X2.T.dot(E_e).dot(X1)) < threshold:
                num_inliers = num_inliers +1
                inliers[k] = True
            else:
                inliers[k] = False

        # print(num_inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            inliers_best = inliers
            points_index_best = points_index

    # print('epoch: ', epochs-1, 'max_inliers: ', max_inliers)
    # print('points_index_best: ', points_index_best)

    # RE-DOING EIGHT POINT ALGO AFTER RANSAC WITH INLIER POINTS
    pts1_inliers = pts1[np.where(inliers_best)]
    pts2_inliers = pts2[np.where(inliers_best)]
    E_best = computeEssentialMtx_8pts(pts1_inliers, pts2_inliers)

    inliers_final = np.zeros((pts1.shape[0], 1))
    for k in range(pts1.shape[0]):
            X2 = np.asarray(  [pts2[k,0], pts2[k,1], 1] )
            X1 = np.asarray(  [pts1[k,0], pts1[k,1], 1] )
            if abs(X2.T.dot(E_best).dot(X1)) < threshold:
                inliers_final[k, 0] = 1

    return E_best, inliers_final

'''
# Recover camera pose from normalized points
    Input:  normalized pts1, Nx3 Matrix
            normalized pts2, Nx3 Matrix
    Output: R, rotation
            t, translation
'''
def poseFromEpipolar(K1, K2, pts1, pts2):

    # EIGHT-POINT ALGORITHM
    # E = computeEssentialMtx_8pts(pts1, pts2)
    E, inliers_final = ransacE(pts1, pts2)

    # CALCULATE M1 and M2
    M1 = np.array([ [ 1,0,0,0 ],
                    [ 0,1,0,0 ],
                    [ 0,0,1,0 ]  ])

    M2_list = candidate_poses(E)

    #  TRIANGULATION
    C1 = K1.dot(M1)

    P_best = np.zeros( (pts1.shape[0],3) )
    M2_best = np.zeros( (3,4) )
    C2_best = np.zeros( (3,4) )
    err_best = np.inf

    error_list = []

    index = 0
    for i in range(M2_list.shape[2]):
        M2 = M2_list[:, :, i]
        C2 = K2.dot(M2)
        P_i, err = triangulate(C1, pts1, C2, pts2)
        error_list.append(err)
        z_list = P_i[:, 2]
        if all( z>0 for z in z_list):
            index = i
            err_best = err
            P_best = P_i
            M2_best = M2
            C2_best = C2

    # print('error_list: ', len(error_list))
    # print('err_best: ', err_best)
    # print('M2_best:\n', M2_best)
    # print('C2_best:\n', C2_best )
    # print('P_best: ', P_best.shape )
    # print('index: ', index)
    R = M2_best[0:3, 0:3]
    t = M2_best[0:3, 3]
    t = t.reshape((3, 1))

    return R, t, inliers_final