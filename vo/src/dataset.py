
import numpy as np 
import cv2
import glob

class Dataset(object):
    def __init__(self, path, name, fps=None, associations=None):
        self.path=path 
        self.name=name 
        # self.type=type    
        self.is_ok = True
        self.fps = fps   
        if fps is not None:       
            self.Ts = 1./fps 
        else: 
            self.Ts = None 
          
        self.timestamps = None 
        self._timestamp = None       # current timestamp if available [s]
        self._next_timestamp = None  # next timestamp if available otherwise an estimate [s]
        
    def isOk(self):
        return self.is_ok

    def getImage(self, frame_id):
        return None 

    def getImage1(self, frame_id):
        return None

    def getDepth(self, frame_id):
        return None        

    def getImageColor(self, frame_id):
        try: 
            img = self.getImage(frame_id)
            if img.ndim == 2:
                return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)     
            else:
                return img             
        except:
            img = None  
            #raise IOError('Cannot open dataset: ', self.name, ', path: ', self.path)        
            print('Cannot open dataset: ', self.name, ', path: ', self.path)
            return img    
        
    def getTimestamp(self):
        return self._timestamp
    
    def getNextTimestamp(self):
        return self._next_timestamp    


class FolderDataset(Dataset):   
    def __init__(self, path, name, fps=None, associations=None): 
        super().__init__(path, name, associations)  
        if fps is None: 
            fps = 10 # default value  
        self.fps = fps 
        print('fps: ', self.fps)  
        self.Ts = 1./self.fps 
        self.skip=1
        self.listing = []    
        self.maxlen = 1000000    
        print('Processing Image Directory Input')
        self.listing = glob.glob(path + '/' + self.name)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        #print('list of files: ', self.listing)
        self.maxlen = len(self.listing)
        self.i = 0        
        if self.maxlen == 0:
            raise IOError('No images were found in folder: ', path)   
        else:
            print('\ttotal number of images: ', self.maxlen)
        self._timestamp = 0.        
            
    def getImage(self, frame_id):
        if self.i == self.maxlen:
            return (None, False)
        image_file = self.listing[self.i]
        img = cv2.imread(image_file)
        self._timestamp += self.Ts
        self._next_timestamp = self._timestamp + self.Ts         
        if img is None: 
            raise IOError('error reading file: ', image_file)               
        # Increment internal counter.
        self.i += 1
        return img


class VideoDataset(Dataset): 
    def __init__(self, path, name, associations=None): 
        super().__init__(path, name, associations)    
        self.filename = path + '/' + name 
        #print('video: ', self.filename)
        self.cap = cv2.VideoCapture(self.filename)
        if not self.cap.isOpened():
            raise IOError('Cannot open movie file: ', self.filename)
        else: 
            print('Processing Video Input')
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            self.Ts = 1./self.fps 
            print('num frames: ', self.num_frames)  
            print('fps: ', self.fps)              
        self.is_init = False   
            
    def getImage(self, frame_id):
        # retrieve the first image if its id is > 0 
        if self.is_init is False and frame_id > 0:
            self.is_init = True 
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        self.is_init = True
        ret, image = self.cap.read()
        #self._timestamp = time.time()  # rough timestamp if nothing else is available 
        self._timestamp = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)*1000)
        self._next_timestamp = self._timestamp + self.Ts 
        if ret is False:
            print('ERROR while reading from file: ', self.filename)
        return image


def dataset_factory(videoDir, videoPath):

    # Folder type
    associations = None
    dataset = FolderDataset(videoDir, videoPath, associations)
    return dataset 

    # video type
    # associations = None
    # dataset = VideoDataset(videoDir, videoPath, associations)