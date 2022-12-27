import numpy as np
import cv2 as cv

class img_processing:
    def __init__(self):
        self.img1 = None
        self.img2 = None
        self.img12_in_one = None
        self.img1_norm = None
        self.img2_norm = None
        self.img_channels = 3 # rgb -> 3 , and gray -> 1

        self.inliers_line_thickness = 3
        self.inliers_threshold = 20 # if more than inliers_threshold, line thickness will be thinner
        self.img_path_stiching2 = 'results/stitch2images.png'

    def load_image(self, img_path1):
        #
        # Read in an image file, errors out if we can't find the file
        #
        img1name = str(img_path1)
        # self.img1 = cv.imread(img1name, 0) # gray image
        self.img1 = cv.imread(img1name) # RGB image
        img_dimension = self.img1.shape

        if self.img1 is not None :
            if len(img_dimension) == 3:
                self.img_channels = 3
                print('An RGB image is loaded. ')
            elif len(img_dimension) == 2:
                self.img_channels = 1
                print('A gray image is loaded. ')
        else:
            print('Unsuccessful Loading! ')

    def load2images(self, img_path1, img_path2):
        #
        # Read in an image file, errors out if we can't find the file
        #
        img1name = str(img_path1)
        img2name = str(img_path2)
        if self.img_channels == 3: # rgb
            self.img1 = cv.imread(img1name) # query image
            self.img2 = cv.imread(img2name) # train image
        elif self.img_channels == 1: # gray
            self.img1 = cv.imread(img1name, 0) # query image
            self.img2 = cv.imread(img2name, 0) # train image
        
        if self.img1 is not None and self.img2 is not None:
            print('2 images are successfully loaded. ')
        else:
            print('Unsuccessful Loading! ')

    def transformImage(self, width_transformed, height_transformed, img_original, transformationMatrix):
        #
        # inverse warping from bilinear interpolation
        #
        # reference
        # https://github.com/makkrnic/inverse-perspective-mapping/blob/master/prototype/transform.py
        #
        img_original = img_original.reshape(img_original.shape[0], img_original.shape[1], -1)
        transformedImage = np.zeros((height_transformed, width_transformed, self.img_channels), dtype=float)
        Hinv = np.linalg.inv(transformationMatrix)
        Hinv /= Hinv[2, 2]
        progress_bar_previous = -10.0
        for y_transformed in range(height_transformed):
            progress_bar = (float(y_transformed)/float(height_transformed-1) * 100)
            if (progress_bar - progress_bar_previous) > 10:
                print("Progress: %.1lf / 100 %%" % progress_bar) 
                progress_bar_previous = progress_bar
            for x_transformed in range(width_transformed):
                point_i_transformed = np.array([[x_transformed], [y_transformed], [1]], dtype=float)
                point_i_transformed /= point_i_transformed[2, 0]
                point_i_original = np.matmul(Hinv, point_i_transformed)
                point_i_original /= point_i_original[2, 0]
                # print ("Point transformed: " +str(point_i_transformed) + ", Point original: " + str (point_i_original) + '\n==============\n')
                xyInt = np.floor(point_i_original)
                xInt = int(xyInt[0])
                yInt = int(xyInt[1])

                if (point_i_original[0] == xInt and point_i_original[1] == yInt):
                    for i in range(self.img_channels):
                        transformedImage[y_transformed, x_transformed, :] = img_original[yInt, xInt, :]
                elif ((point_i_original[0] != xInt or point_i_original[1] != yInt)
                    and point_i_original[0] >= 0 and point_i_original[1] >= 0
                    and point_i_original[0] + 1 < img_original.shape[1]
                    and point_i_original[1] + 1 < img_original.shape[0]):
                    # print ("Interpolating (%f, %f) at (%f, %f)" % (point_i_original[0], point_i_original[1], point_i_transformed[0], point_i_transformed[1]))
                    dx = point_i_original[0] - xInt
                    dy = point_i_original[1] - yInt

                    w = img_original[yInt:(yInt+2), xInt:(xInt+2), :]
                    w = w.reshape(w.shape[0], w.shape[1], -1)
                    w_dist = (np.array([[(1-dx) * (1-dy), dx * (1-dy)], [(1-dx) * dy, dx * dy]])).reshape(2,2)
                    for i in range(self.img_channels):
                        point_rgb_i = np.vdot(w[:, :, i], w_dist)
                        transformedImage[y_transformed, x_transformed, i] = point_rgb_i
                
                else:
                    print ("Not interpolating (%f, %f)" % (point_i_original[0], point_i_original[1]))
                    
        print('The image has been completely transformed!')
        
        return transformedImage

    def stitch2images(self, saveImg=False):
        # Stitch 2 images together
        rows1 = self.img1.shape[0]
        cols1 = self.img1.shape[1]
        rows2 = self.img2.shape[0]
        cols2 = self.img2.shape[1]
        img_channels = 3 # rgb data
        if len(self.img1.shape) == 2: # gray image
            img_channels = 1 # gray image

        out = np.zeros((max([rows1,rows2]), cols1+cols2, 3), dtype='uint8')

        if img_channels == 3: # rgb data
            # Place the first image to the left
            out[:rows1,:cols1,:] = np.dstack([self.img1[:, :, 0], self.img1[:, :, 1], self.img1[:, :, 2]])

            # Place the next image to the right of it
            out[:rows2,cols1:cols1+cols2,:] = np.dstack([self.img2[:, :, 0], self.img2[:, :, 1], self.img2[:, :, 2]])

        elif img_channels == 1: # gray data
            # Place the first image to the left
            out[:rows1,:cols1,:] = np.dstack([self.img1, self.img1, self.img1])

            # Place the next image to the right of it
            out[:rows2,cols1:cols1+cols2,:] = np.dstack([self.img2, self.img2, self.img2])  

        self.img12_in_one = out

        if saveImg:
            cv.imwrite(self.img_path_stiching2, out)

    def drawFeatures(self, matches, keypoint_1, keypoint_2, img_path = 'img1.png', inliers = None, drawOutliers = True):
        # Create a new output image that concatenates the two images together
        # # This draws matches and optionally a set of inliers in a different color
        #
        # Note: this function is modified from the following github source:
        # https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py
        #

        self.stitch2images()
        cols1 = self.img1.shape[1]

        # For each pair of points we have between both images
        # draw circles, then connect a line between them
        for mat in matches:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns, y - rows
            (x1,y1) = keypoint_1[img1_idx].pt
            (x2,y2) = keypoint_2[img2_idx].pt

            inlier = False

            if inliers is not None:
                NumOfInliers = len(inliers)
                self.inliers_line_thickness = 3
                if NumOfInliers > self.inliers_threshold:
                    self.inliers_line_thickness = 1
                for i in inliers:
                    if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                        inlier = True

            # Draw a small circle at both co-ordinates
            cv.circle(self.img12_in_one, (int(x1),int(y1)), 4, (255, 0, 0), 1)
            cv.circle(self.img12_in_one, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        cv.imwrite(img_path, self.img12_in_one)

    def drawMatches(self, matches, keypoint_1, keypoint_2, img_path = 'img1.png', inliers = None, drawOutliers = True):
        # Create a new output image that concatenates the two images together
        # # This draws matches and optionally a set of inliers in a different color
        #
        # Note: this function is modified from the following github source:
        # https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py
        #

        self.stitch2images()
        cols1 = self.img1.shape[1]

        # For each pair of points we have between both images
        # draw circles, then connect a line between them
        for mat in matches:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns, y - rows
            (x1,y1) = keypoint_1[img1_idx].pt
            (x2,y2) = keypoint_2[img2_idx].pt

            inlier = False

            if inliers is not None:
                NumOfInliers = len(inliers)
                self.inliers_line_thickness = 3
                if NumOfInliers > self.inliers_threshold:
                    self.inliers_line_thickness = 1
                for i in inliers:
                    if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                        inlier = True

            # Draw a small circle at both co-ordinates
            cv.circle(self.img12_in_one, (int(x1),int(y1)), 4, (255, 0, 0), 1)
            cv.circle(self.img12_in_one, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

            # Draw a line in between the two points, draw inliers if we have them
            if inliers is None: # without knowledge of inliers
                cv.line(self.img12_in_one, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
            elif inliers is not None and inlier: # draw inliers
                cv.line(self.img12_in_one, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1 * self.inliers_line_thickness)
            elif drawOutliers and inliers is not None: # draw non-inlier points
                cv.line(self.img12_in_one, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        cv.imwrite(img_path, self.img12_in_one)