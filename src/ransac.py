import numpy as np
import random

from src.homography_estimation import homography_estimation

class ransac:
    def __init__(self, maxInliers_thresh=0.7, geometricDistance_thresh=5, NumberOfInterestPoints=4, ransac_max_iterations=1000):
        self.maxInliers_thresh = maxInliers_thresh
        self.geometricDistance_thresh = geometricDistance_thresh # 5
        self.NumberOfInterestPoints = NumberOfInterestPoints
        self.ransac_max_iterations = ransac_max_iterations # 1000
        self.camera_mtx = homography_estimation()
        self.homography_alg = 'DLT'
        self.save_homography_DLT_log = 'results/homography_DLT.txt'
        self.save_homography_NDLT_log = 'results/homography_NDLT.txt'

    def geometricDistance(self, correspondence, h):
        #
        # Calculate the geometric distance between estimated points and original points
        #
        # reference
        # https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py
        #
        try: 
            p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
            estimatep2 = np.dot(h, p1)
            estimatep2 = (1 / estimatep2.item(2)) * estimatep2

            p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
            error = p2 - estimatep2

            return np.linalg.norm(error)

        except:
            return 100 # max number


    def save_log(self, finalH, inliers_length):
        #
        # Save homography matrix
        #
        txt_path = self.save_homography_DLT_log
        if self.homography_alg == 'NDLT':
            txt_path = self.save_homography_NDLT_log
        f = open(txt_path, 'w')
        f.write("Final homography: \n" + str(finalH) + "\n")
        f.write("Final inliers count: " + str(inliers_length))
        f.close()
        print('log info saved!')

    def run_basic(self, corr, sub_corr, homography_alg, NumberOfInterestPoints_basic=4):
        #
        # Runs through basic ransac algorithm, creating homographies from random 4-correspondences
        #
        self.homography_alg = homography_alg
        maxInliers = []
        finalH = None
        finalCorr = None

        for k in range(self.ransac_max_iterations):
            # return a list of selected 4 numbers, without duplicates.
            idx_interested = random.sample(range(len(sub_corr)), NumberOfInterestPoints_basic)
            randomCorr = [] # find 4 random points to calculate a homography
            for j in range(1, NumberOfInterestPoints_basic):
                if j == 1:
                    randomCorr = np.vstack((sub_corr[idx_interested[0]], sub_corr[idx_interested[1]]))
                else:
                    randomCorr = np.vstack((randomCorr, sub_corr[idx_interested[j]]))

            # call the homography function on those points
            if self.homography_alg == 'DLT':
                self.camera_mtx.DLT(randomCorr)
            elif self.homography_alg == 'NDLT':
                self.camera_mtx.normalized_DLT(randomCorr)

            inliers = []
            for i in range(len(corr)):
                if self.geometricDistance(corr[i], self.camera_mtx.H) < self.geometricDistance_thresh:
                    inliers.append(corr[i])

            if len(inliers) > len(maxInliers):
                maxInliers = inliers
                finalH = self.camera_mtx.H
                finalCorr = randomCorr

            if len(maxInliers) > (len(corr) * self.maxInliers_thresh):
                break
            
            if k % 100 == 0:
                print("iter: ", k, " / ", self.ransac_max_iterations, " correspondence#: ", len(corr), " NumInliers/Max: ", len(inliers), " / ", len(maxInliers))

        return finalH, finalCorr, maxInliers

    def run(self, corr, homography_alg):
        #
        # Runs through ransac algorithm, creating homographies from random n-correspondences
        #

        if self.NumberOfInterestPoints <= 4:
            finalH, finalCorr, maxInliers = self.run_basic(corr, corr, homography_alg, 4)
        else: 
            # First we get inliers from random 4-correspondences
            print("4-correspondences")
            _H, Corr4, inliers_corr4 = self.run_basic(corr, corr, homography_alg, 4)
            # Then, we use these inliers to estimate n-correspondence, where n > 4. 
            inliers_corr4 = np.array(inliers_corr4)
            inliers_corr4 = inliers_corr4.reshape(inliers_corr4.shape[0], -1)
            inliers_corr4 = np.matrix(inliers_corr4)
            while len(inliers_corr4) <= self.NumberOfInterestPoints:
                print("RANSAC 4-correspondences again")
                _H, Corr4, inliers_corr4 = self.run_basic(corr, corr, homography_alg, 4)
            print("n-correspondences")
            finalH, finalCorr, maxInliers = self.run_basic(corr, inliers_corr4, homography_alg, self.NumberOfInterestPoints)

        print("Homography =\n", finalH)
        print("inliers count: ", len(maxInliers))

        self.save_log(finalH, len(maxInliers))

        return finalH, finalCorr, maxInliers