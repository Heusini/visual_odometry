import cv2 as cv
import numpy as np
from camera import Camera
from camera_pose import plot_plotly
from helpers import load_images
from enum import Enum

def pnp():
    pass

class DataSetEnum(Enum):
    KITTI = "kitti"
    PARKING = "parking"
    MALAGA = "malaga"

if __name__ == "__main__":
    
    dataset = DataSetEnum.KITTI
    stride = 4
    steps = 5
    
    K_kitty = np.array(
        [
            [7.188560000000e02, 0, 6.071928000000e02],
            [0, 7.188560000000e02, 1.852157000000e02],
            [0, 0, 1],
        ]
    )
    K_parking = np.array([[331.37, 0, 320], [0, 369.568, 240], [0, 0, 1]])

    if dataset == DataSetEnum.KITTI:
        K = K_kitty
        imgs = load_images("data/kitti/05/image_0/", start=0, end=stride*steps)
    elif dataset == DataSetEnum.PARKING:
        K = K_parking
        imgs = load_images("data/parking/images/", start=0, end=stride*steps)

    cams = []

    for step in range(steps):
        img = imgs[int(step*stride)]
        cam = Camera(step, K, img)
        cam.calculate_features()
        cams.append(cam)

    Ps = []
    for step in range(steps - 1):
        cam_i = cams[step]
        cam_j = cams[step+1]
        (m1, m2), (_, _) = cam_i.calculate_matches(cam_j)
        P = cam_i.process_next_frame(cam_j, m1, m2)
        Ps.append(P)

    plot_plotly(Ps, cams)