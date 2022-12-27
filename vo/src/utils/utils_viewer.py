############################
#                          #
#    Visualization         #
#                          #
# Author: David Wang       #
# Created on Nov. 14, 2022 #
#                          #
############################

import numpy as np
import cv2

from vo.src.utils.mplot_thread import viewOpen3d, Mplot3d, Mplot2d

class viewer_set():
    def __init__(self, UseOpen3D, is_draw_3d, is_draw_matched_points_count, is_draw_traj_img):
        self.UseOpen3D = UseOpen3D  
        self.is_draw_3d = is_draw_3d # False
        self.is_draw_matched_points_count = is_draw_matched_points_count # False 
        self.is_draw_traj_img = is_draw_traj_img
        self.traj3Dviewer, self.err_plt, self.matched_points_plt = None, None, None

        self.draw_scale = 1
        traj_img_size = 800
        self.half_traj_img_size = int(0.5*traj_img_size)
        self.traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
        
    def init(self):

        
        if self.is_draw_3d:
            if self.UseOpen3D:
                self.traj3Dviewer = viewOpen3d()
            else:
                self.traj3Dviewer = Mplot3d(title='3D trajectory')
        
        if self.is_draw_matched_points_count:
            
            self.matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')

        # return self.traj3Dviewer, self.err_plt, self.matched_points_plt

    def draw_2d_trajectory(self, img_id, vo_traj3d_est):
        x, y, z = vo_traj3d_est[-1]

        if self.is_draw_traj_img:      # draw 2D trajectory (on the plane xz)
            draw_x, draw_y = int(self.draw_scale*x) + self.half_traj_img_size, self.half_traj_img_size - int(self.draw_scale*z)
            cv2.circle(self.traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
            # write text on traj_img
            cv2.rectangle(self.traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
            text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
            cv2.putText(self.traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
            # show      
            cv2.imshow('Trajectory', self.traj_img)


    def draw_3d_trajectory(self, img_id, vo):
        if self.is_draw_3d:           # draw 3d trajectory 
            if self.UseOpen3D:
                self.traj3Dviewer.drawTraj(vo.cur_R, vo.cur_t_bias)
            else:
                # self.traj3Dviewer.drawTraj(vo.traj3d_gt,'ground truth',color='r',marker='.')
                self.traj3Dviewer.drawTraj(vo.traj3d_est,'estimated',color='g',marker='.')
                self.traj3Dviewer.refresh()

  
    def draw_matched_points(self, img_id, vo):
        if self.is_draw_matched_points_count:
            matched_kps_signal = [img_id, vo.num_matched_kps]
            inliers_signal = [img_id, vo.num_inliers]                    
            self.matched_points_plt.draw(matched_kps_signal,'# matches',color='b')
            self.matched_points_plt.draw(inliers_signal,'# inliers',color='g')                    
            self.matched_points_plt.refresh()  

    def draw_live_image(self, vo_draw_img):
        # draw camera image 
        cv2.imshow('Camera', vo_draw_img)

    def visualize_process(self, img_id, vo):
        self.draw_live_image(vo.draw_img)
        if img_id > 0:
            self.draw_2d_trajectory(img_id, vo.traj3d_est)
            self.draw_3d_trajectory(img_id, vo)
            self.draw_matched_points(img_id, vo)
        
        # press 'q' to exit!
        return (cv2.waitKey(1) & 0xFF == ord('q'))

    def destroy_all_viewers(self):

        print('\nPress any key to exit ...')
        cv2.waitKey(0)

        if self.is_draw_traj_img:
            print('saving map.png')
            cv2.imwrite('map.png', self.traj_img)
        if self.is_draw_3d:
            self.traj3Dviewer.quit()
        if self.is_draw_matched_points_count:
            self.matched_points_plt.quit()
                    
        cv2.destroyAllWindows()