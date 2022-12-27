import cv2
import numpy as np

class correspondences_estimation:
    def __init__(self, img1, img2, feature_alg):
        self.img1 = img1
        self.img2 = img2
        self.valid_images = False
        if self.img1 is not None and self.img2 is not None:
            self.valid_images = True
        self.keypoint_1, self.descriptor_1, self.keypoint_2, self.descriptor_2, self.matches, self.correspondences = None, None, None, None, None, None
        self.img_path_keypoint1 = 'results/orb_keypoints_1.png'
        self.img_path_keypoint2 = 'results/orb_keypoints_2.png'
        self.img_path_matching = 'results/match_orb.png'
        self.img_path_matching_inlier = 'results/match_inliers_DLT.png'
        self.inliers_line_thickness = 3
        self.inliers_threshold = 20 # if more than inliers_threshold, line thickness will be thinner
        if feature_alg == 's': # sift
            self.feature_alg = cv2.SIFT_create()   # opencv-python==4.5.1.48
        elif feature_alg == 'o': # ORB
            self.feature_alg = cv2.ORB_create()   # opencv-python==4.5.1.48
 
    #
    # Runs sift algorithm to find features
    #
    def findFeatures(self, img, img_path):
        print("Finding Features...")
        keypoints, descriptors = self.feature_alg.detectAndCompute(img, None)

        img = cv2.drawKeypoints(img, keypoints, 0)
        cv2.imwrite(img_path, img)
        print(img_path + " has " + str(len(keypoints)) + " keypoints.")

        return keypoints, descriptors

    #
    # Matches features given a list of keypoints, descriptors, and images
    #
    def matchFeatures(self):
        print("Matching Features...")
        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_L2, True)
        # Match descriptors
        self.matches = matcher.match(self.descriptor_1, self.descriptor_2)

    def get_pairs(self):
        correspondenceList = []
        
        self.keypoint_1, self.descriptor_1 = self.findFeatures(self.img1, self.img_path_keypoint1)
        self.keypoint_2, self.descriptor_2 = self.findFeatures(self.img2, self.img_path_keypoint2)
        keypoints = [self.keypoint_1, self.keypoint_2]
        self.matchFeatures()
        
        for match in self.matches:
            (x1, y1) = keypoints[0][match.queryIdx].pt
            (x2, y2) = keypoints[1][match.trainIdx].pt
            correspondenceList.append([x1, y1, x2, y2])

        self.correspondences = np.matrix(correspondenceList)