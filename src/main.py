from enum import Enum
import cv2 as cv
import numpy as np
from camera import Camera
from utils.image_loader import PathLoader

# Constants
init_bandwidth = 4

class DataSetEnum(Enum):
    KITTI = "kitti"
    PARKING = "parking"
    MALAGA = "malaga"

def main():
    
    dataset = DataSetEnum.KITTI

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
        path = "data/kitti/05/image_0/"
    elif dataset == DataSetEnum.PARKING:
        K = K_parking
        path = "data/parking/images/"

    path_loader = PathLoader(path, start=0, stride=init_bandwidth)

    cam_0 = Camera(0, K, next(path_loader))
    cam_1 = Camera(1, K, next(path_loader))
    cam_0.calculate_features()
    cam_1.calculate_features()

    (m1, m2), (_, _) = cam_0.calculate_matches(cam_1)
    P = cam_0.calculate_points_in_world(cam_1, m1, m2)
    

if __name__ == '__main__':
    main()