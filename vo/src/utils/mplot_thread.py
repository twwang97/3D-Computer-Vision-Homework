
import time 
import sys 
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

import multiprocessing as mp 
from multiprocessing import Process, Queue, Lock, RLock, Value


kVerbose = False 
kSetDaemon = True   # from https://docs.python.org/3/library/threading.html#threading.Thread.daemon
                    # The entire Python program exits when no alive non-daemon threads are left.

kUseFigCanvasDrawIdle = True  
kPlotSleep = 0.04

# global lock for drawing with matplotlib 
mp_lock = RLock()

if kUseFigCanvasDrawIdle:
    plt.ion()
    

# use mplotlib figure to draw in 2d dynamic data
class Mplot2d:
    def __init__(self, xlabel='', ylabel='', title=''):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title 

        self.data = None 
        self.got_data = False 

        self.axis_computed = False 
        self.xlim = [float("inf"),float("-inf")]
        self.ylim = [float("inf"),float("-inf")]    

        self.key = Value('i',0)
        self.is_running = Value('i',1)

        self.handle_map = {}        

        self.queue = Queue()
        self.vp = Process(target=self.drawer_thread, args=(self.queue,mp_lock,self.key,self.is_running,))
        self.vp.daemon = kSetDaemon
        self.vp.start()

    def quit(self):
        self.is_running.value = 0
        self.vp.join(timeout=5)

    def drawer_thread(self, queue, lock, key, is_running):  
        self.init(lock) 
        #print('starting drawer_thread')
        while is_running.value == 1:
            #print('drawer_refresh step')
            self.drawer_refresh(queue, lock)                                    
            if kUseFigCanvasDrawIdle:               
                time.sleep(kPlotSleep) 
        print(mp.current_process().name,'closing fig ', self.fig)  
        plt.close(self.fig)              

    def drawer_refresh(self, queue, lock):            
        while not queue.empty():      
            self.got_data = True           
            self.data = queue.get()          
            xy_signal, name, color, marker = self.data 
            #print(mp.current_process().name,"refreshing : signal ", name)            
            if name in self.handle_map:
                handle = self.handle_map[name]
                handle.set_xdata(np.append(handle.get_xdata(), xy_signal[0]))
                handle.set_ydata(np.append(handle.get_ydata(), xy_signal[1]))                
            else: 
                handle, = self.ax.plot(xy_signal[0], xy_signal[1], c=color, marker=marker, label=name)    
                self.handle_map[name] = handle  
        #print(mp.current_process().name,"got data: ", self.got_data) 
        if self.got_data is True:                   
            self.plot_refresh(lock)

    def on_key_press(self, event):
        #print(mp.current_process().name,"key event pressed...", self._key)     
        self.key.value = ord(event.key) # conver to int 
        
    def on_key_release(self, event):
        #print(mp.current_process().name,"key event released...", self._key)             
        self.key.value = 0  # reset to no key symbol
        
    def get_key(self):
        return chr(self.key.value)            

    def init(self, lock):    
        lock.acquire()      
        if kVerbose:
            print(mp.current_process().name,"initializing...") 
        self.fig = plt.figure()
        if kUseFigCanvasDrawIdle:
            self.fig.canvas.draw_idle() 
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)       
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)               
        #self.ax = self.fig.gca(projection='3d')
        #self.ax = self.fig.gca()
        self.ax = self.fig.add_subplot(111)   
        if self.title != '':
            self.ax.set_title(self.title) 
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)	   
        self.ax.grid()		
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        #self.refresh()     
        lock.release()

    def setAxis(self):		                     
        self.ax.legend()
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        if not kUseFigCanvasDrawIdle:
            self.fig.canvas.draw()
        self.fig.canvas.flush_events()        

    def draw(self, xy_signal, name, color='r', marker='.'):    
        if self.queue is None:
            return
        self.queue.put((xy_signal, name, color, marker))

    def updateMinMax(self, np_signal):
        xmax,ymax = np.amax(np_signal,axis=0)
        xmin,ymin = np.amin(np_signal,axis=0)        
        cx = 0.5*(xmax+xmin)
        cy = 0.5*(ymax+ymin) 
        if False: 
            # update maxs       
            if xmax > self.xlim[1]:
                self.xlim[1] = xmax 
            if ymax > self.ylim[1]:
                self.ylim[1] = ymax                   
            # update mins
            if xmin < self.xlim[0]:
                self.xlim[0] = xmin   
            if ymin < self.ylim[0]:
                self.ylim[0] = ymin        
        # make axis actually squared
        if True:
            smin = min(xmin,ymin)                                            
            smax = max(xmax,ymax)            
            delta = 0.5*(smax - smin)
            self.xlim = [cx-delta,cx+delta]
            self.ylim = [cy-delta,cy+delta]   
        self.axis_computed = True   

    def plot_refresh(self, lock):
        if kVerbose:        
            print(mp.current_process().name,"refreshing ", self.title)          
        lock.acquire()         
        self.setAxis()
        if not kUseFigCanvasDrawIdle:        
            plt.pause(kPlotSleep)
        lock.release()

    # fake 
    def refresh(self):
        pass 


# use mplotlib figure to draw in 3D trajectories 
class Mplot3d:
    def __init__(self, title=''):
        self.title = title 

        self.data = None  
        self.got_data = False 

        self.axis_computed = False 
        self.xlim = [float("inf"),float("-inf")]
        self.ylim = [float("inf"),float("-inf")]
        self.zlim = [float("inf"),float("-inf")]        

        self.handle_map = {}     
        
        self.key = Value('i',0)
        self.is_running = Value('i',1)         

        self.queue = Queue()
        self.vp = Process(target=self.drawer_thread, args=(self.queue,mp_lock, self.key, self.is_running,))
        self.vp.daemon = kSetDaemon
        self.vp.start()

    def quit(self):
        self.is_running.value = 0
        self.vp.join(timeout=5)     
        
    def drawer_thread(self, queue, lock, key, is_running):  
        self.init(lock) 
        while is_running.value == 1:
            self.drawer_refresh(queue, lock)   
            if kUseFigCanvasDrawIdle:               
                time.sleep(kPlotSleep)    
        print(mp.current_process().name,'closing fig ', self.fig)     
        plt.close(self.fig)                                 

    def drawer_refresh(self, queue, lock):            
        while not queue.empty():    
            self.got_data = True  
            self.data = queue.get()  
            traj, name, color, marker = self.data         
            np_traj = np.asarray(traj)        
            if name in self.handle_map:
                handle = self.handle_map[name]
                self.ax.collections.remove(handle)
            self.updateMinMax(np_traj)
            handle = self.ax.scatter3D(np_traj[:, 0], np_traj[:, 1], np_traj[:, 2], c=color, marker=marker)
            handle.set_label(name)
            self.handle_map[name] = handle     
        if self.got_data is True:               
            self.plot_refresh(lock)          

    def on_key_press(self, event):
        #print(mp.current_process().name,"key event pressed...", self._key)     
        self.key.value = ord(event.key) # conver to int 
        
    def on_key_release(self, event):
        #print(mp.current_process().name,"key event released...", self._key)             
        self.key.value = 0  # reset to no key symbol
        
    def get_key(self):
        return chr(self.key.value) 
    
    def init(self, lock):
        lock.acquire()      
        if kVerbose:
            print(mp.current_process().name,"initializing...") 
        self.fig = plt.figure()
        if kUseFigCanvasDrawIdle:
            self.fig.canvas.draw_idle()         
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)       
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)             
        self.ax = self.fig.gca(projection='3d')
        if self.title != '':
            self.ax.set_title(self.title)     
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')		   		

        self.setAxis()
        lock.release() 

    def setAxis(self):		
        #self.ax.axis('equal')   # this does not work with the new matplotlib 3    
        if self.axis_computed:	
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)  
            self.ax.set_zlim(self.zlim)                             
        self.ax.legend()
        #We need to draw *and* flush
        if not kUseFigCanvasDrawIdle:
            self.fig.canvas.draw()
        self.fig.canvas.flush_events()            

    def drawTraj(self, traj, name, color='r', marker='.'):
        if self.queue is None:
            return
        self.queue.put((traj, name, color, marker))

    def updateMinMax(self, np_traj):
        xmax,ymax,zmax = np.amax(np_traj,axis=0)
        xmin,ymin,zmin = np.amin(np_traj,axis=0)        
        cx = 0.5*(xmax+xmin)
        cy = 0.5*(ymax+ymin)
        cz = 0.5*(zmax+zmin) 
        if False: 
            # update maxs       
            if xmax > self.xlim[1]:
                self.xlim[1] = xmax 
            if ymax > self.ylim[1]:
                self.ylim[1] = ymax 
            if zmax > self.zlim[1]:
                self.zlim[1] = zmax                         
            # update mins
            if xmin < self.xlim[0]:
                self.xlim[0] = xmin   
            if ymin < self.ylim[0]:
                self.ylim[0] = ymin        
            if zmin < self.zlim[0]:
                self.zlim[0] = zmin     
        # make axis actually squared
        if True:
            #smin = min(self.xlim[0],self.ylim[0],self.zlim[0])                                            
            #smax = max(self.xlim[1],self.ylim[1],self.zlim[1])
            smin = min(xmin,ymin,zmin)                                            
            smax = max(xmax,ymax,zmax)            
            delta = 0.5*(smax - smin)
            self.xlim = [cx-delta,cx+delta]
            self.ylim = [cy-delta,cy+delta]
            self.zlim = [cz-delta,cz+delta]      
        self.axis_computed = True   

    def plot_refresh(self, lock):
        if kVerbose:        
            print(mp.current_process().name,"refreshing ", self.title)          
        lock.acquire()          
        self.setAxis()
        if not kUseFigCanvasDrawIdle:        
            plt.pause(kPlotSleep)      
        lock.release()

    # fake 
    def refresh(self):
        pass         


class viewOpen3d(): 
    def __init__(self):
        line_scale_wh = 0.8
        line_scale_z = 1.0
        self.camera_skeleton_pts = [[0, 0, 0], [line_scale_wh, line_scale_wh, line_scale_z], 
                                                            [-line_scale_wh, line_scale_wh, line_scale_z], 
                                                            [-line_scale_wh, -line_scale_wh, line_scale_z], 
                                                            [line_scale_wh, -line_scale_wh, line_scale_z]]
        # self.camera_skeleton_pts = [[0, 0, 0], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]]
        self.camera_skeleton_lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        rgb_color_code = np.array([0, 0, 0.6])
        self.camera_skeleton_colors = np.tile(rgb_color_code, (8, 1))
        self.camera_skeleton_zeros3 = np.zeros(3)
        self.camera_skeleton_scale = 0.5

        self.queue = mp.Queue()
        self.p = mp.Process(target=self.process_frames, args=(self.queue, ))
        self.p.start()
    
        

    def draw_camera(self, rot, pos):
        model = o3d.geometry.LineSet()
        model.points = o3d.utility.Vector3dVector(self.camera_skeleton_pts)
        model.lines = o3d.utility.Vector2iVector(self.camera_skeleton_lines)
        # model.colors = o3d.utility.Vector3dVector(self.camera_skeleton_colors)
        model.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], 
                                                  [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
        model.scale(self.camera_skeleton_scale, self.camera_skeleton_zeros3)
        model.rotate(rot)
        model.translate(pos)
        return model

    def process_frames(self, queue_p):

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        keep_running = True
        while keep_running:
            try:
                R, t = queue_p.get(block=False)
                if R is not None:                    
                    model = self.draw_camera(R, t)
                    self.vis.add_geometry(model)
                    pass
            except: 
                pass
            keep_running = keep_running and self.vis.poll_events()

        self.vis.destroy_window()
        self.p.join()
        

    def drawTraj(self, _r_mtx, _t_vec):
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
        # _r_mtx = np.transpose(_r_mtx)
        # _t_vec = - _r_mtx @ _t_vec
        self.queue.put((_r_mtx, _t_vec))

    def quit(self):
        a = 1

'''
class viewOpen3d(): 
    def __init__(self):

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

    def draw_camera(self, rot, pos):
        model = o3d.geometry.LineSet()
        model.points = o3d.utility.Vector3dVector(
            [[0, 0, 0], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]])
        model.lines = o3d.utility.Vector2iVector(
            [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
        color = np.array([1, 0, 0])
        model.colors = o3d.utility.Vector3dVector(np.tile(color, (8, 1)))
        model.scale(0.5, np.zeros(3))
        model.rotate(rot)
        model.translate(pos)
        return model

    def drawTraj(self, R, t):
        cameraSkeleton = self.draw_camera(R, t)
        self.vis.add_geometry(cameraSkeleton)
        return self.vis.poll_events()

'''