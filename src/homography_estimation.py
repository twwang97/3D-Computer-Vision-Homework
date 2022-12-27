import numpy as np

class homography_estimation:
    def __init__(self):
        self.H = None # Homography matrix 3 x 3
    
    def calculateHomography(self, correspondences):
        #
        # Computers a homography from n-correspondences
        #
        list_A = []
        for corr in correspondences:
            p1 = np.matrix([corr.item(0), corr.item(1), 1])
            p2 = np.matrix([corr.item(2), corr.item(3), 1])

            row1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                  p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
            row2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                  p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
            
            list_A.append(row1)
            list_A.append(row2)

        mtx_A = np.matrix(list_A)

        # SVD composition
        u, s, v = np.linalg.svd(mtx_A)

        # reshape the min singular value into a 3 by 3 matrix
        self.H = np.reshape(v[8], (3, 3))

        # normalize and now we have H
        self.H = (1 / self.H.item(8)) * self.H

        return self.H

    def normalize_points(self, points):
        #
        # normalize points
        #
        points = np.array(points)
        mean = np.mean(points, axis=0)

        translation = np.array([[1., 0., -1.*mean[0]], [0., 1., -1.*mean[1]], [0., 0., 1.]])
        scale_rate = np.mean(
            np.sqrt(np.sum((points[:]-[mean[0], mean[1]])**2, axis=1))) 
        scale = np.eye(3)
        scale[0, 0], scale[1, 1] = 1./scale_rate, 1./scale_rate
        points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

        transformation = np.matmul(scale, translation)
        new_points = np.matmul(transformation, points.T).T[:, 0:2]

        return transformation, new_points

    def normalize_correspondences(self, correspondences):
        #
        # normalize 2 sets of points
        #
        transformation_1, points1_norm = self.normalize_points(correspondences[:, 0:2])
        transformation_2, points2_norm = self.normalize_points(correspondences[:, 2:4])
        points_norm = np.concatenate((points1_norm, points2_norm), axis=1)
        return points_norm, transformation_1, transformation_2
        
    def DLT(self, correspondences):
        # 
        # Direct Linear Transform (DLT)
        #
        self.calculateHomography(correspondences)
        return self.H

    def normalized_DLT(self, correspondences):
        # 
        # Normalized Direct Linear Transform (Normalized DLT)
        #
        correspondences_norm, T1, T2 = self.normalize_correspondences(correspondences)
        H_prime = self.calculateHomography(correspondences_norm)
        self.H = np.matmul(np.linalg.inv(T2), np.matmul(H_prime, T1))
        self.H /= self.H[2, 2]
        return self.H