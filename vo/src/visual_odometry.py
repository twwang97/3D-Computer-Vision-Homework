############################
#                          #
#    Visual Odometry       #
#                          #
# Author: David Wang       #
# Created on Nov. 14, 2022 #
#                          #
# modified from PYSLAM.    #
# Please refer to          #
# luigifreda's github      #
#                          #
############################

import numpy as np 
import cv2
from enum import Enum
import math # sqrt

from vo.src.feature.feature_tracker import FeatureTrackerTypes, FeatureTrackingResult
from vo.src.utils.utils_geom import poseRt
from vo.src.timer import TimerFps
from vo.src.pose_estimator import poseFromEpipolar

class VoStage(Enum):
    NO_IMAGES_YET   = 0     # no image received 
    GOT_FIRST_IMAGE = 1     # got first image, we can proceed in a normal way (match current image with previous image)
    
kVerbose = True     
kMinNumFeature = 2000
kRansacThresholdNormalized = 0.0003  # metric threshold used for normalized image coordinates 
kRansacThresholdPixels = 0.1         # pixel threshold used for image coordinates 
kAbsoluteScaleThreshold = 0.1        # absolute translation scale; it is also the minimum translation norm for an accepted motion 
kRansacProb = 0.999
kUseGroundTruthScale = True 


# This class is a first start to understand the basics of inter frame feature tracking and camera pose estimation.
# It combines the simplest VO ingredients without performing any image point triangulation or 
# windowed bundle adjustment. At each step $k$, it estimates the current camera pose $C_k$ with respect to the previous one $C_{k-1}$. 
# The inter frame pose estimation returns $[R_{k-1,k},t_{k-1,k}]$ with $||t_{k-1,k}||=1$. 
# With this very basic approach, you need to use a ground truth in order to recover a correct inter-frame scale $s$ and estimate a 
# valid trajectory by composing $C_k = C_{k-1} * [R_{k-1,k}, s t_{k-1,k}]$. 
class VisualOdometry(object):
    def __init__(self, cam, feature_tracker, is_transformed_grayscale, UsePoseNewMethod):
        self.UsePoseNewMethod = False
        if UsePoseNewMethod == 'mymethod':
            self.UsePoseNewMethod = True
        self.stage = VoStage.NO_IMAGES_YET
        self.cam = cam
        self.cur_image = None   # current image
        self.prev_image = None  # previous/reference image
        self.is_transformed_grayscale = False

        self.kps_ref = None  # reference keypoints 
        self.des_ref = None # refeference descriptors 
        self.kps_cur = None  # current keypoints 
        self.des_cur = None # current descriptors 

        self.cur_R = np.eye(3,3) # current rotation 
        self.cur_t = np.zeros((3,1)) # current translation 
        self.cur_t_bias = np.zeros((3,1)) # current translation from the origin

        # self.trueX, self.trueY, self.trueZ = None, None, None
        # self.groundtruth = groundtruth
        
        self.feature_tracker = feature_tracker
        self.track_result = None 

        self.mask_match = None # mask of matched keypoints used for drawing 
        self.draw_img = None 

        self.init_history = True 
        self.poses = []              # history of poses
        self.t0_est = None           # history of estimated translations      
        # self.t0_gt = None            # history of ground truth translations (if available)
        self.traj3d_est = []         # history of estimated translations centered w.r.t. first one
        # self.traj3d_gt = []          # history of estimated ground truth translations centered w.r.t. first one  

        self.num_matched_kps = None    # current number of matched keypoints  
        self.num_inliers = None        # current number of inliers 

        self.timer_verbose = True # set this to True if you want to print timings 
        self.timer_main = TimerFps('VO', is_verbose = self.timer_verbose)
        self.timer_pose_est = TimerFps('PoseEst', is_verbose = self.timer_verbose)
        self.timer_feat = TimerFps('Feature', is_verbose = self.timer_verbose)


    def computeFundamentalMatrix(self, kps_ref, kps_cur):
            F, mask = cv2.findFundamentalMat(kps_ref, kps_cur, cv2.FM_RANSAC, param1=kRansacThresholdPixels, param2=kRansacProb)
            if F is None or F.shape == (1, 1):
                # no fundamental matrix found
                raise Exception('No fundamental matrix found')
            elif F.shape[0] > 3:
                # more than one matrix found, just pick the first
                F = F[0:3, 0:3]
            return np.matrix(F), mask 	

    def removeOutliersByMask(self, mask): 
        if mask is not None:    
            n = self.kpn_cur.shape[0]     
            mask_index = [ i for i,v in enumerate(mask) if v > 0]    
            self.kpn_cur = self.kpn_cur[mask_index]           
            self.kpn_ref = self.kpn_ref[mask_index]           
            if self.des_cur is not None: 
                self.des_cur = self.des_cur[mask_index]        
            if self.des_ref is not None: 
                self.des_ref = self.des_ref[mask_index]  
            if kVerbose:
                print('removed ', n-self.kpn_cur.shape[0],' outliers')                
    
    # Absolute Orientation Problem: 
    # Since the translation is up to a scale, 
    # we need to recover the translation vector. 
    # https://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho2-08-absolute-orient.pptx.pdf
    def rescale_translation_factor(self, kp_ref_u, kp_cur_u):
        mean_ref = np.mean(self.kpn_ref, axis=0)
        mean_cur = np.mean(self.kpn_cur, axis=0)
        ref_squared = 0.0
        cur_squared = 0.0
        for i in range(self.kpn_ref.shape[0]):
            kp_ref_u_0mean_i = self.kpn_ref[i] - mean_ref
            kp_cur_u_0mean_i = self.kpn_cur[i] - mean_cur
            ref_squared += np.dot(kp_ref_u_0mean_i, kp_ref_u_0mean_i)
            cur_squared += np.dot(kp_cur_u_0mean_i, kp_cur_u_0mean_i)
        rescale_factor = math.sqrt(cur_squared/ref_squared)
        rescale_factor = min(max(rescale_factor, 0.5), 2.0) # bound the scale in the range of [0.5 2.0]
        return rescale_factor
        
    # fit essential matrix E with RANSAC such that:  p2.T * E * p1 = 0  where  E = [t21]x * R21
    # out: [Rrc, trc]   (with respect to 'ref' frame) 
    # N.B.1: trc is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with previous estimated poses)
    # N.B.2: this function has problems in the following cases: [see Hartley/Zisserman Book]
    # - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie on a ruled quadric 
    # - degenerate motions such a pure rotation (a sufficient parallax is required) or an infinitesimal viewpoint change (where the translation is almost zero)
    # N.B.3: the five-point algorithm (used for estimating the Essential Matrix) seems to work well in the degenerate planar cases [Five-Point Motion Estimation Made Easy, Hartley]
    # N.B.4: as reported above, in case of pure rotation, this algorithm will compute a useless fundamental matrix which cannot be decomposed to return the rotation 
    def estimatePose(self, kps_ref, kps_cur):	
        kp_ref_u = self.cam.undistort_points(kps_ref)	
        kp_cur_u = self.cam.undistort_points(kps_cur)	        
        self.kpn_ref = self.cam.unproject_points(kp_ref_u)
        self.kpn_cur = self.cam.unproject_points(kp_cur_u)
        if self.UsePoseNewMethod:
            R, t, self.mask_match = poseFromEpipolar(self.cam.K, self.cam.K, self.kpn_cur, self.kpn_ref)
        else: # from opencv
            E, self.mask_match = cv2.findEssentialMat(self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)
            _, R, t, mask = cv2.recoverPose(E, self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.))   
            # F, self.mask_match = self.computeFundamentalMatrix(kp_cur_u, kp_ref_u)
            # E = self.cam.K.T @ F @ self.cam.K    # E = K.T * F * K 
            t = t * self.rescale_translation_factor(kp_ref_u, kp_cur_u)
        return R,t  # Rrc, trc (with respect to 'ref' frame) 		

    def processFirstFrame(self):
        # only detect on the current image 
        self.kps_ref, self.des_ref = self.feature_tracker.detectAndCompute(self.cur_image)
        # convert from list of keypoints to an array of points 
        self.kps_ref = np.array([x.pt for x in self.kps_ref], dtype=np.float32) 
        self.draw_img = self.drawFeatureTracks(self.cur_image)

    def processFrame(self, frame_id):
        # track features 
        self.timer_feat.start()
        self.track_result = self.feature_tracker.track(self.prev_image, self.cur_image, self.kps_ref, self.des_ref)
        self.timer_feat.refresh()
        # estimate pose 
        self.timer_pose_est.start()
        R, t = self.estimatePose(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)     
        self.timer_pose_est.refresh()
        # update keypoints history  
        self.kps_ref = self.track_result.kps_ref
        self.kps_cur = self.track_result.kps_cur
        self.des_cur = self.track_result.des_cur 
        self.num_matched_kps = self.kpn_ref.shape[0] 
        self.num_inliers =  np.sum(self.mask_match)
        if kVerbose:        
            print('# matched points: ', self.num_matched_kps, ', # inliers: ', self.num_inliers)      
        self.cur_t = self.cur_t + self.cur_R.dot(t) 
        self.cur_R = self.cur_R.dot(R)       
        # draw image         
        self.draw_img = self.drawFeatureTracks(self.cur_image) 
        # check if we have enough features to track otherwise detect new ones and start tracking from them (used for LK tracker) 
        if (self.feature_tracker.tracker_type == FeatureTrackerTypes.LK) and (self.kps_ref.shape[0] < self.feature_tracker.num_features): 
            self.kps_cur, self.des_cur = self.feature_tracker.detectAndCompute(self.cur_image)           
            self.kps_cur = np.array([x.pt for x in self.kps_cur], dtype=np.float32) # convert from list of keypoints to an array of points   
            if kVerbose:     
                print('# new detected points: ', self.kps_cur.shape[0])                  
        self.kps_ref = self.kps_cur
        self.des_ref = self.des_cur
        self.updateHistory()           
        

    def track(self, img, frame_id):
        if kVerbose:
            print('..................................')
            print('frame: ', frame_id) 

        if self.is_transformed_grayscale:
            # convert image to gray if needed    
            if img.ndim > 2:
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)             
            # check coherence of image size with camera settings 
            assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        
        self.cur_image = img
        # manage and check stage 
        if(self.stage == VoStage.GOT_FIRST_IMAGE):
            self.processFrame(frame_id)
        elif(self.stage == VoStage.NO_IMAGES_YET):
            self.processFirstFrame()
            self.stage = VoStage.GOT_FIRST_IMAGE            
        self.prev_image = self.cur_image    
        # update main timer (for profiling)
        self.timer_main.refresh()  
  

    def drawFeatureTracks(self, img, reinit = False):
        if self.is_transformed_grayscale:
            draw_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        else:
            draw_img = img
        num_outliers = 0        
        if(self.stage == VoStage.GOT_FIRST_IMAGE):            
            if reinit:
                for p1 in self.kps_cur:
                    a,b = p1.ravel()
                    cv2.circle(draw_img,(a,b),1, (0,255,0),-1)                    
            else:    
                for i,pts in enumerate(zip(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)):
                    drawAll = False # set this to true if you want to draw outliers 
                    if self.mask_match[i] or drawAll:
                        p1, p2 = pts 
                        a,b = p1.astype(int).ravel()
                        c,d = p2.astype(int).ravel()
                        cv2.line(draw_img, (a,b),(c,d), (0,255,0), 1)
                        cv2.circle(draw_img,(a,b),1, (0,0,255),-1)   
                    else:
                        num_outliers+=1
            if kVerbose:
                print('# outliers: ', num_outliers)     
        return draw_img            

    def updateHistory(self):
        if self.init_history == True:
            # starting translation 
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  
            self.init_history = False 
        if self.t0_est is not None:             
            # the estimated traj starts at 0
            self.cur_t_bias = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   
            self.traj3d_est.append(self.cur_t_bias)
            self.poses.append(poseRt(self.cur_R, self.cur_t_bias))   
