
import cv2 as cv
import numpy as np

from src.image_processing import img_processing


WINDOW_NAME = 'window_p2'

def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])  

def show_img_in_a_while(img):
    cv.namedWindow(WINDOW_NAME)
    while True:
        cv.imshow(WINDOW_NAME, img)
        key = cv.waitKey(20) % 0xFF
        if key == 27: break # exit when pressing ESC
    cv.destroyAllWindows()

# Click 4 corners of the book
def track_mouse_4clicks(img1_path):

    img_class = img_processing()
    img_class.load2images(img1_path, 'images/example_book.png')

    # to show the image smaller than your computer screen

    # img-1 # shape = (586, 480, 3)
    scale_percent = 1 # ratio = 0 ~ 1
    max_img_length = 700
    if max(img_class.img1.shape[0], img_class.img1.shape[1]) > max_img_length:
        scale_percent = max_img_length / max(img_class.img1.shape[0], img_class.img1.shape[1]) # ratio of the original size
    width = int(img_class.img1.shape[1] * scale_percent)
    height = int(img_class.img1.shape[0] * scale_percent)
    dim = (width, height)   
    img_class.img1 = cv.resize(img_class.img1, dim, interpolation = cv.INTER_AREA) # resize image

    # img-2 (example)
    scale_percent = 0.5
    width = int(img_class.img2.shape[1] * scale_percent)
    height = int(img_class.img2.shape[0] * scale_percent)
    dim = (width, height)
    img_class.img2 = cv.resize(img_class.img2, dim, interpolation = cv.INTER_AREA) # resize image

    img_class.stitch2images(True)

    # track mouse
    numberOfCorners = 4
    points_set= []
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_set])
    while len(points_set) != numberOfCorners:
        img_ = img_class.img12_in_one.copy()
        for i, p in enumerate(points_set):
            # draw points on img_
            cv.circle(img_, tuple(p), 2, (0, 255, 0), int(max(img_class.img1.shape[0], img_class.img1.shape[1]) / 70))
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27: break # exit when pressing ESC

    cv.destroyAllWindows()
    print('{} Points added: '.format(len(points_set)))
    print(points_set)

    return np.array(points_set), img_class.img1
