from enum import Enum
import cv2 as cv
import numpy as np
from state import FrameSate
from utils.path_loader import PathLoader
from pnp import pnp

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
    path_iter = iter(path_loader)

    # initialize first two states, corresponding features are computed with initialization
    state_0 = FrameSate(0, next(path_iter))
    state_1 = FrameSate(init_bandwidth, next(path_iter))

    # computes landmarks and pose for the first two frames
    pnp(state_0, state_1, K)
    

if __name__ == '__main__':
    main()