##############################
#                            #
#       Mesh Displayer       #
#                            #
# Render the .obj file,      #
# e.g., 3D mesh, by using    #
# the pygame library.        #
#                            #
# This project is created by #
# Lachlan Cox and modified by#
# David Wang.                #
#                            #
##############################

# Github Source
# https://github.com/lcox74/Py3D

from PyDisplayer.GUI_task import GUIpygame

def main(): 
    # obj_filePath = './model3D/Cube.obj'
    obj_filePath = './model3D/Map.obj'
    WindowName = 'Object Viewer'
    displayer = GUIpygame(WindowName, 800, 600, meshPath=obj_filePath)
    displayer.run()

if __name__ == '__main__': 
    main()