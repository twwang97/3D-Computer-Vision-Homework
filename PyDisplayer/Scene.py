from PyDisplayer.Camera import Camera

from PyDisplayer.Entities.Cube import Cube

class Scene:
	def __init__ (self, surface, width, height, meshPath="./Model/Map.obj"):
		self.entities = []

		self.surface = surface
		self.cam = Camera(width, height)

		cube = Cube(meshPath, 0, 2, 3)
		self.entities.append(cube)

	def draw (self):
		self.cam.update()

		for e in self.entities:
			e.draw(self.surface, self.cam)

	def key_pressed (self, keys):
		self.cam.key_pressed(keys)
		for entity in self.entities:
			entity.key_pressed(keys)
