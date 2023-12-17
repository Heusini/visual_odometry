import numpy as np
from camera_pose import Camera


def pnp():
    pass


if __name__ == "__main__":
    K_parking = np.array([[331.37, 0, 320], [0, 369.568, 240], [0, 0, 1]])
    img1 = cv.imread("data/parking/images/img_00000.png")
    img2 = cv.imread("data/parking/images/img_00003.png")
    K = K_parking
    cam1 = Camera(1, np.eye(3), np.zeros((3, 1)), None)
    cam2 = Camera(2, None, None, None)

    matches = {}
    matches[cam1.id] = {}
    matches[cam1.id][cam2.id] = []

    cam1.calculate_features(img1)
    cam2.calculate_features(img2)
