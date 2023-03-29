
import pygame

from PyDisplayer.Scene import Scene

class GUIpygame:
    def __init__ (self, title, _width, _height, meshPath="./Model/Map.obj"):
        pygame.init()
        self.surface = pygame.display.set_mode((_width, _height))
        pygame.display.set_caption(title)

        self.clock = pygame.time.Clock()
        self.fps = 60

        self.running = True
        self.resolution = 2

        self.Scenes = Scene(self.surface, _width, _height, meshPath)

    def run (self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            keys = pygame.key.get_pressed()
            if len(keys):
                self.Scenes.key_pressed(keys)

            self.surface.fill((10, 10, 10))
            self.Scenes.draw()
            print("tick {:d} | fps {:.4f}".format(self.clock.tick(self.fps), self.clock.get_fps()))

            pygame.display.update()
        pygame.quit()