import cv2 as cv
import numpy as np
from camera import Camera
from camera_pose import plot_plotly


def pnp():
    pass


if __name__ == "__main__":
    K_parking = np.array([[331.37, 0, 320], [0, 369.568, 240], [0, 0, 1]])
    img1 = cv.imread("data/parking/images/img_00000.png")
    img2 = cv.imread("data/parking/images/img_00003.png")
    K = K_parking
    cam1 = Camera(1, K, img1, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    cam2 = Camera(2, K, img2)

    matches = {}
    matches[cam1.id] = {}
    matches[cam1.id][cam2.id] = []

    cam1.calculate_features()
    cam2.calculate_features()

    (m1, m2), (_, _) = cam1.calculate_matches(cam2)

    P = cam1.calculate_points_in_world(cam2, m1, m2)

    plot_plotly(P, [cam1, cam2])
