import open3d as o3d
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

def load_point_cloud(points3D_df):

    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd

def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B
    return axes

def get_transform_mtx(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat

def initialize_cube_pose():
    global R_euler, t, scale
    R_euler = np.array([0, 0, 0]).astype(float)
    t = np.array([0, 0, 0]).astype(float)
    scale = 1.0

def update_cube():
    global cube, cube_vertices, R_euler, t, scale
    
    transform_mat = get_transform_mtx(R_euler, t, scale)
    
    transform_vertices = (transform_mat @ np.concatenate([
                            cube_vertices.transpose(), 
                            np.ones([1, cube_vertices.shape[0]])
                            ], axis=0)).transpose()

    cube.vertices = o3d.utility.Vector3dVector(transform_vertices)
    cube.compute_vertex_normals()
    cube.paint_uniform_color([1, 0.706, 0])
    vis.update_geometry(cube)
    print('vertices')
    print(np.asarray(cube.vertices))

def toggle_key_shift(vis, action, mods):
    global shift_pressed
    if action == 1: # key down
        shift_pressed = True
    elif action == 0: # key up
        shift_pressed = False
    return True

def update_tx(vis):
    global t, shift_pressed
    t[0] += -0.01 if shift_pressed else 0.01
    update_cube()

def update_ty(vis):
    global t, shift_pressed
    t[1] += -0.01 if shift_pressed else 0.01
    update_cube()

def update_tz(vis):
    global t, shift_pressed
    t[2] += -0.01 if shift_pressed else 0.01
    update_cube()

def update_rx(vis):
    global R_euler, shift_pressed
    R_euler[0] += -1 if shift_pressed else 1
    update_cube()

def update_ry(vis):
    global R_euler, shift_pressed
    R_euler[1] += -1 if shift_pressed else 1
    update_cube()

def update_rz(vis):
    global R_euler, shift_pressed
    R_euler[2] += -1 if shift_pressed else 1
    update_cube()

def update_scale(vis):
    global scale, shift_pressed
    scale += -0.05 if shift_pressed else 0.05
    update_cube()

def set_keyCallback():
    global shift_pressed, vis

    # set key callback
    shift_pressed = False
    vis.register_key_action_callback(340, toggle_key_shift)
    vis.register_key_action_callback(344, toggle_key_shift)
    vis.register_key_callback(ord('A'), update_tx)
    vis.register_key_callback(ord('S'), update_ty)
    vis.register_key_callback(ord('D'), update_tz)
    vis.register_key_callback(ord('Z'), update_rx)
    vis.register_key_callback(ord('X'), update_ry)
    vis.register_key_callback(ord('C'), update_rz)
    vis.register_key_callback(ord('V'), update_scale)

    print('[Keyboard usage]')
    print('Translate along X-axis\tA / Shift+A')
    print('Translate along Y-axis\tS / Shift+S')
    print('Translate along Z-axis\tD / Shift+D')
    print('Rotate    along X-axis\tZ / Shift+Z')
    print('Rotate    along Y-axis\tX / Shift+X')
    print('Rotate    along Z-axis\tC / Shift+C')
    print('Scale                 \tV / Shift+V')

def initialize_o3d_visualization():
    global vis
    # just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mtx(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)

    set_keyCallback()

def update_o3d_drawing():
    global vis, R_euler, t, scale, cube
    vis.run()
    vis.destroy_window()

    print('Rotation matrix:\n{}'.format(R.from_euler('xyz', R_euler, degrees=True).as_matrix()))
    print('Translation vector:\n{}'.format(t))
    print('Scale factor: {}'.format(scale))

    np.save('cube_transform_mat.npy', get_transform_mtx(R_euler, t, scale))
    np.save('cube_vertices.npy', np.asarray(cube.vertices))
    print(np.asarray(cube.vertices))

def main():
    global R_euler, t, scale, cube, cube_vertices, vis, shift_pressed

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # load point cloud
    points3D_df = pd.read_pickle("data/points3D.pkl")
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)

    # load axes
    axes = load_axes()
    vis.add_geometry(axes)

    # load cube
    cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    cube_vertices = np.asarray(cube.vertices).copy()
    vis.add_geometry(cube)

    initialize_cube_pose()
    update_cube()
    initialize_o3d_visualization()
    update_o3d_drawing()

if __name__ == '__main__':
    main()
