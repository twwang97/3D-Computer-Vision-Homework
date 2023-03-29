import pygame
import numpy as np
import math

class Camera:
	def __init__ (self, width, height, fov = 90, near = 0.1, far = 1000):
		self.position = (0, 0, 0)
		self.rotation = (0, 0, 0)

		self.lookDir = (0, 0, 1)
		self.up = (0, 1, 0)

		self.forward = (0, 0, 1)

		self.height = height
		self.width = width
		self.fov = fov
		self.far = far
		self.near = near

		self.Aspect = self.height / self.width
		self.FOVRad = 1 / math.tan(fov * 0.5 * math.pi / 180)

	def update (self):
		self.forward = self.AddVector(self.position, self.lookDir)

	def GetProjection (self):
		a = np.identity(4)
		a[0][0] = self.Aspect * self.FOVRad
		a[1][1] = self.FOVRad
		a[2][2] = self.far / (self.far - self.near)
		a[3][2] = (-self.far * self.near) / (self.far - self.near)
		a[2][3] = 1
		a[3][3] = 0
		return a

	def GetLookAt (self):
		return self.LookAt(self.position, self.forward, self.up)

	def LookAt (self, pos, target, up):
		newForward = self.NormaliseVector(self.SubVector(target, pos))

		a = self.MulVector(newForward, self.DotProduct(up, newForward))
		newUp = self.NormaliseVector(self.SubVector(up, a))

		newRight = self.CrossProduct(newUp, newForward)

		mat = np.identity(4)
		mat[0][0] = newRight[0]
		mat[0][1] = newUp[0]
		mat[0][2] = newForward[0]
		mat[1][0] = newRight[1]
		mat[1][1] = newUp[1]
		mat[1][2] = newForward[1]
		mat[2][0] = newRight[2]
		mat[2][1] = newUp[2]
		mat[2][2] = newForward[2]
		mat[3][0] = self.DotProduct(self.MulVector(pos, -1), newRight)
		mat[3][1] = self.DotProduct(self.MulVector(pos, -1), newUp)
		mat[3][2] = self.DotProduct(self.MulVector(pos, -1), newForward)
		return mat

	def key_pressed(self, keys):
		if keys[pygame.K_w]:
			a = self.MulVector(self.lookDir, 1/10)
			self.position = (self.position[0] + a[0], self.position[1] + a[1], self.position[2] + a[2])
		if keys[pygame.K_s]:
			a = self.MulVector(self.lookDir, -1/10)
			self.position = (self.position[0] + a[0], self.position[1] + a[1], self.position[2] + a[2])
		if keys[pygame.K_a]:
			a = self.MulVector(self.CrossProduct(self.up, self.lookDir), -1/10)
			self.position = (self.position[0] + a[0], self.position[1] + a[1], self.position[2] + a[2])
		if keys[pygame.K_d]:
			a = self.MulVector(self.CrossProduct(self.up, self.lookDir), 1/10)
			self.position = (self.position[0] + a[0], self.position[1] + a[1], self.position[2] + a[2])

		if keys[pygame.K_LEFT]:
			self.rotation = (self.rotation[0], self.rotation[1] + 1/10, self.rotation[2])
			self.lookDir = self.MatrixMultiplyToVector(self.lookDir, self.MakeRotationY(1/10))
		if keys[pygame.K_RIGHT]:
			self.rotation = (self.rotation[0], self.rotation[1] - 1/10, self.rotation[2])
			self.lookDir = self.MatrixMultiplyToVector(self.lookDir, self.MakeRotationY(-1/10))

	def CrossProduct (self, a, b):
		x = a[1] * b[2] - a[2] * b[1]
		y = a[2] * b[0] - a[0] * b[2]
		z = a[0] * b[1] - a[1] * b[0]
		return (x, y, z)
	def DotProduct (self, a, b):
		return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

	def NormaliseVector (self, a):
		l = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
		return (a[0] / l, a[1] / l, a[2] / l)
	def SubVector (self, a, b):
		return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
	def AddVector (self, a, b):
		return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
	def MulVector (self, a, b):
		return (a[0] * b, a[1] * b, a[2] * b)

	def MakeTranslation (self, x, y, z):
		mat = np.identity(4)
		mat[3][0] = x
		mat[3][1] = y
		mat[3][2] = z
		return mat

	def MakeRotationX (self, a):
		mat = np.identity(4)
		mat[1][1] = math.cos(a)
		mat[1][2] = math.sin(a) 
		mat[2][1] = -math.sin(a)
		mat[2][2] = math.cos(a)
		return mat

	def MakeRotationY (self, a):
		mat = np.identity(4)
		mat[0][0] = math.cos(a)
		mat[0][2] = math.sin(a) 
		mat[2][0] = -math.sin(a)
		mat[2][2] = math.cos(a)
		return mat

	def MakeRotationZ (self, a):
		mat = np.identity(4)
		mat[0][0] = math.cos(a)
		mat[0][1] = math.sin(a) 
		mat[1][0] = -math.sin(a)
		mat[1][1] = math.cos(a)
		return mat

	def MatrixMultiplyToVector (self, v, m):
		tempx = v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0] + m[3][0]
		tempy = v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1] + m[3][1]
		tempz = v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2] + m[3][2]
		w = v[0] * m[0][3] + v[1] * m[1][3] + v[2] * m[2][3] + m[3][3]

		if not (w == 0):
			return (tempx / w, tempy / w, tempz / w)

		return (v[0], v[1], v[2])
