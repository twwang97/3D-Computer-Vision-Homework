import pygame, os, math
import numpy as np

class Triangle:
	def __init__(self, a = 0, b = 0, c = 0, ua = 0, ub = 0, uc = 0):
		self.p = [a, b, c] # (0, 0, 0) * 3
		self.u = [ua, ub, uc] # (0, 0) * 3
		self.colour = (255, 255, 255)

		self.key = []

	def __str__ (self):
		return ("[{0}, {1}, {2}]".format(self.p[0], self.p[1], self.p[2]))

class Mesh:
	def __init__(self):
		self.tris = []

	def LoadObj (self, filename):
		obj_file = open(filename)

		temp_verts = []
		temp_uvs = []

		for line in obj_file:
			line = line.strip()

			try:
				if len(line) == 0:
					continue
				elif line[0] == '#' or line == '\n':
					continue
			except:
				print("An exception occurred") 
				print()
				print(type(line), len(line))

			if len(line) == 0:
					continue
			elif line[0] == '#' or line == '\n':
				continue
			elif line[0] == 'v':
				if line[1] == ' ':
					temp = line.split(' ')
					temp_verts.append((float(temp[1]), float(temp[2]), float(temp[3])))
				elif line[1] == 't':
					temp = line.split(' ')
					temp_uvs.append((float(temp[1]), float(temp[2]))) # u
			elif line[0] == 'f':
				temp = line.split(' ')

				temp_face = 0

				if '/' in line:
					nline = line.split(' ')
					temp_face = Triangle(temp_verts[int(nline[1].split('/')[0]) - 1],
										 temp_verts[int(nline[2].split('/')[0]) - 1],
										 temp_verts[int(nline[3].split('/')[0]) - 1],
										 temp_uvs[int(nline[1].split('/')[1]) - 1],
										 temp_uvs[int(nline[2].split('/')[1]) - 1],
										 temp_uvs[int(nline[3].split('/')[1]) - 1])
				else:
					nline = line.split(' ')
					temp_face = Triangle(temp_verts[int(nline[1]) - 1],
										 temp_verts[int(nline[2]) - 1],
										 temp_verts[int(nline[3]) - 1])

				self.tris.append(temp_face)
		obj_file.close()

class Entity:
	def __init__ (self, tag, x = 0, y = 0, z = 0):
		self.tag = tag

		self.position = (x, y, z)
		self.mesh = Mesh()

		self.fTheta = 0

	def update (self):
		pass

	def key_pressed (self, key):
		pass

	def draw (self, surface, cam):

		matRotZ = self.MakeRotationZ(self.fTheta)
		matRotX = self.MakeRotationX(180)
		matTrans = self.MakeTranslation(self.position[0], self.position[1], self.position[2])

		TrianglesToRaster = []

		matView = cam.GetLookAt()
		matProj = cam.GetProjection()

		matWorld = matRotZ * matRotX * matTrans

		for tri in self.mesh.tris:
			triProjected = Triangle()
			triTransformed = Triangle()
			triViewed = Triangle()

			triTransformed.p[0] = self.MatrixMultiplyToVector(tri.p[0], matWorld)
			triTransformed.p[1] = self.MatrixMultiplyToVector(tri.p[1], matWorld)
			triTransformed.p[2] = self.MatrixMultiplyToVector(tri.p[2], matWorld)

			line1 = self.SubVector(triTransformed.p[1], triTransformed.p[0])
			line2 = self.SubVector(triTransformed.p[2], triTransformed.p[0])

			normal = self.NormaliseVector(self.CrossProduct(line1, line2))

			#if (normal[2] < 0):
			if (self.DotProduct(normal, self.SubVector(triTransformed.p[0], cam.position)) < 0):

				lightDirection = self.NormaliseVector((0.2, 0.3, -0.7))

				dotProd = abs(self.DotProduct(normal, lightDirection))
				col = (tri.colour[0] * dotProd, tri.colour[1] * dotProd, tri.colour[2] * dotProd)

				triViewed.p[0] = self.MatrixMultiplyToVector(triTransformed.p[0], matView)
				triViewed.p[1] = self.MatrixMultiplyToVector(triTransformed.p[1], matView)
				triViewed.p[2] = self.MatrixMultiplyToVector(triTransformed.p[2], matView)

				ClippedData = self.Triangle_ClipAgainstPlane((0, 0, cam.near), (0, 0, 1), triViewed)

				for x in range(ClippedData[0]):
					# 3D --> 2D
					triProjected.p[0] = self.MatrixMultiplyToVector(ClippedData[x + 1].p[0], matProj)
					triProjected.p[1] = self.MatrixMultiplyToVector(ClippedData[x + 1].p[1], matProj)
					triProjected.p[2] = self.MatrixMultiplyToVector(ClippedData[x + 1].p[2], matProj)

					triProjected.p[0] = ((triProjected.p[0][0] + 1) * 0.5 * cam.width, (triProjected.p[0][1] + 1) * 0.5 * cam.height, triProjected.p[0][2])
					triProjected.p[1] = ((triProjected.p[1][0] + 1) * 0.5 * cam.width, (triProjected.p[1][1] + 1) * 0.5 * cam.height, triProjected.p[1][2])
					triProjected.p[2] = ((triProjected.p[2][0] + 1) * 0.5 * cam.width, (triProjected.p[2][1] + 1) * 0.5 * cam.height, triProjected.p[2][2])
					triProjected.colour = col

					TrianglesToRaster.append(triProjected)

		# sort z back to front
		TrianglesToRaster.sort(key=lambda x: (x.p[0][2] + x.p[1][2] + x.p[2][2]) / 3, reverse=True)

		for triToRaster in TrianglesToRaster:
			listTriangles = []

			listTriangles.append(triToRaster)
			nNewTriangles = 1

			for p in range(4):
				nTrisToAdd = [0, None, None]
				while nNewTriangles > 0:
					test = listTriangles[0]
					listTriangles.pop()
					nNewTriangles -= 1
					if p == 0:
						nTrisToAdd = self.Triangle_ClipAgainstPlane((0, 0, 0), (0, 1, 0), test)
					elif p == 1:
						nTrisToAdd = self.Triangle_ClipAgainstPlane((0, cam.height - 1, 0), (0, -1, 0), test)
					elif p == 2:
						nTrisToAdd = self.Triangle_ClipAgainstPlane((0, 0, 0), (1, 0, 0), test)
					elif p == 3:
						nTrisToAdd = self.Triangle_ClipAgainstPlane((cam.width - 1, 0, 0), (-1, 0, 0), test)

					for w in range(nTrisToAdd[0]):
						listTriangles.append(nTrisToAdd[w + 1])
				nNewTriangles = len(listTriangles)

			for tri in listTriangles:
				pygame.draw.polygon(surface, tri.colour,
						[
							[tri.p[0][0], tri.p[0][1]],
							[tri.p[1][0], tri.p[1][1]],
							[tri.p[2][0], tri.p[2][1]]
						])

				# comment the code below to make the lines disappear
				pygame.draw.lines(surface, (255, 255, 255), True,
					[
						[tri.p[0][0], tri.p[0][1]],
						[tri.p[1][0], tri.p[1][1]],
						[tri.p[2][0], tri.p[2][1]]
					])


	def VectorIntersectPlane (self, plane_p, plane_n, lineStart, lineEnd):
		_plane_n = self.NormaliseVector(plane_n)
		plane_d = -self.DotProduct(_plane_n, plane_p)
		ad = self.DotProduct(lineStart, _plane_n)
		bd = self.DotProduct(lineEnd, _plane_n)
		t = (-plane_d - ad) / (bd - ad)
		lineStartToEnd = self.SubVector(lineEnd, lineStart)
		lineToIntersect = self.MulVector(lineStartToEnd, t)
		return self.AddVector(lineStart, lineToIntersect)

	def Triangle_ClipAgainstPlane (self, plane_p, plane_n, in_tri): # [Num, Tri1, Tri2]
		returnArray = [0, None, None]

		_plane_n = self.NormaliseVector(plane_n)

		def dist (p):
			n = self.NormaliseVector(p)
			return (_plane_n[0] * p[0] + _plane_n[1] * p[1] + _plane_n[2] * p[2] - self.DotProduct(_plane_n, plane_p))

		inside_points = [None] * 3
		outside_points = [None] * 3

		nInsidePointCount = 0
		nOutsidePointCount = 0

		d0 = dist(in_tri.p[0])
		d1 = dist(in_tri.p[1])
		d2 = dist(in_tri.p[2])

		if d0 >= 0:
			inside_points[nInsidePointCount] = in_tri.p[0]
			nInsidePointCount += 1
		else:
			outside_points[nOutsidePointCount] = in_tri.p[0]
			nOutsidePointCount += 1

		if d1 >= 0:
			inside_points[nInsidePointCount] = in_tri.p[1]
			nInsidePointCount += 1
		else:
			outside_points[nOutsidePointCount] = in_tri.p[1]
			nOutsidePointCount += 1

		if d2 >= 0:
			inside_points[nInsidePointCount] = in_tri.p[2]
			nInsidePointCount += 1
		else:
			outside_points[nOutsidePointCount] = in_tri.p[2]
			nOutsidePointCount += 1

		if nInsidePointCount == 0:
			return returnArray
		if nInsidePointCount == 3:
			returnArray[0] = 1
			returnArray[1] = in_tri
			return returnArray
		if nInsidePointCount == 1 and nOutsidePointCount == 2:
			tempTriangle1 = Triangle()
			tempTriangle1.colour = in_tri.colour
			#tempTriangle1.colour = (0, 0, 255)
			
			tempTriangle1.p[0] = inside_points[0]
			tempTriangle1.p[1] = self.VectorIntersectPlane(plane_p, plane_n, inside_points[0], outside_points[0])
			tempTriangle1.p[2] = self.VectorIntersectPlane(plane_p, plane_n, inside_points[0], outside_points[1])

			returnArray[0] = 1
			returnArray[1] = tempTriangle1
			return returnArray
		if nInsidePointCount == 2 and nOutsidePointCount == 1:
			tempTriangle1 = Triangle()
			tempTriangle2 = Triangle()

			tempTriangle1.colour = in_tri.colour
			tempTriangle2.colour = in_tri.colour
			#tempTriangle1.colour = (255, 0, 0)
			#tempTriangle2.colour = (0, 255, 0)

			tempTriangle1.p[0] = inside_points[0]
			tempTriangle1.p[1] = inside_points[1]
			tempTriangle1.p[2] = self.VectorIntersectPlane(plane_p, plane_n, inside_points[0], outside_points[0])

			tempTriangle2.p[0] = inside_points[1]
			tempTriangle2.p[1] = tempTriangle1.p[2]
			tempTriangle2.p[2] = self.VectorIntersectPlane(plane_p, plane_n, inside_points[1], outside_points[0])

			returnArray[0] = 2
			returnArray[1] = tempTriangle1
			returnArray[2] = tempTriangle2
			return returnArray


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

