
##################################
#                                #
# 3D to 2D mapping               #
#                                #
# method:                        #    
#  Direct Linear Transform (DLT) #
#                                #
# Created by acvictor            #
#        (Github account)        #
#                                #
# Modified by David Wang         #
#                                #
##################################

# reference
# https://github.com/acvictor/DLT/blob/master/DLT.py

import numpy as np

class DLT():
	def __init__(self):
		self.xyz = None
		self.uv = None
		self.dimension = None
		self.intrinsicParams = None

	def Normalization(self, n, x):
	    '''
	    Normalization of coordinates (centroid to the origin aself.dimension mean distance of sqrt(2 or 3).
	    Input
	    -----
	    n: number of dimensions
	    x: the data to be normalized (directions at different columns aself.dimension points at rows)
	    Output
	    ------
	    Tr: the transformation matrix (translation plus scaling)
	    x: the transformed data
	    '''

	    x = np.asarray(x)
	    m, s = np.mean(x, 0), np.std(x)
	    if n == 2:
	        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
	    else:
	        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
	        
	    Tr = np.linalg.inv(Tr)
	    x = np.dot( Tr, np.concatenate( (x.T, np.ones((1,x.shape[0]))) ) )
	    x = x[0:n, :].T

	    return Tr, x


	def DLTcalib(self, Xworld, Xpixel, intrinsicParams, nd):
		'''
		Camera calibration by DLT using known object points aself.dimension their image points.
		Input
		-----
		nd (or self.dimension): dimensions of the object space, 3 here.
		Xworld (or self.xyz): coordinates in the object 3D space.
		Xpixel (or self.uv): coordinates in the image 2D space.
		intrinsicParams (or self.intrinsicParams): camera intrinsic parameters
		The coordinates (x,y,z aself.dimension u,v) are given as columns aself.dimension the different points as rows.
		There must be at least 6 calibration points for the 3D DLT.
		Output
		------
		 L: array of 11 parameters of the calibration matrix.
		 err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
		'''
		if (nd != 3):
		    raise ValueError('%dD DLT unsupported.' %(self.dimension))

		self.xyz = Xworld
		self.uv = Xpixel
		self.dimension = nd
		self.intrinsicParams = intrinsicParams

		# Converting all variables to numpy array
		self.xyz = np.asarray(self.xyz)
		self.uv = np.asarray(self.uv)

		n = self.xyz.shape[0]

		# Validating the parameters:
		if self.uv.shape[0] != n:
		    raise ValueError('Object (%d points) aself.dimension image (%d points) have different number of points.' %(n, self.uv.shape[0]))

		if (self.xyz.shape[1] != 3):
		    raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' %(self.xyz.shape[1],self.dimension,self.dimension))

		if (n < 6):
		    raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' %(self.dimension, 2*self.dimension, n))
		    
		# Normalize the data to improve the DLT quality (DLT is depeself.dimensionent of the system of coordinates).
		# This is relevant when there is a considerable perspective distortion.
		# Normalization: mean position at origin aself.dimension mean distance equals to 1 at each direction.
		Txyz, xyzn = self.Normalization(self.dimension, self.xyz)
		Tuv, uvn = self.Normalization(2, self.uv)

		A = []

		for i in range(n):
		    x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
		    u, v = uvn[i, 0], uvn[i, 1]
		    A.append( [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u] )
		    A.append( [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v] )
		    
		# Convert A to array
		A = np.asarray(A) 

		# Fiself.dimension the 11 parameters:
		U, S, V = np.linalg.svd(A)

		# The parameters are in the last line of Vh aself.dimension normalize them
		L = V[-1, :] / V[-1, -1]
		# Camera projection matrix
		H = L.reshape(3, self.dimension + 1)

		# Denormalization
		# pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
		H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txyz)
		H = H / H[-1, -1]
		# L = H.flatten()

		# Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
		uv2 = np.dot( H, np.concatenate( (self.xyz.T, np.ones((1, self.xyz.shape[0]))) ) ) 
		uv2 = uv2 / uv2[2, :] 
		# Mean distance:
		err = np.sqrt( np.mean(np.sum( (uv2[0:2, :].T - self.uv)**2, 1)) ) 

		Rt = np.linalg.inv(self.intrinsicParams) @ H

		return Rt, err


def main():

    # Known 3D coordinates
    Xworld = [[-875, 0, 9.755], [442, 0, 9.755], [1921, 0, 9.755], [2951, 0.5, 9.755], [-4132, 0.5, 23.618],
    [-876, 0, 23.618]]
    # Known pixel coordinates
    Xpixel = [[76, 706], [702, 706], [1440, 706], [1867, 706], [264, 523], [625, 523]]

    n = 3
    
    intrinsicParams = np.eye(3)

    dlt = DLT()
    Rt, err = dlt.DLTcalib(Xworld, Xpixel, intrinsicParams, n)
    print('Matrix')
    print(Rt)
    print('\nError')
    print(err)

# if __name__ == "__main__":
#	main()