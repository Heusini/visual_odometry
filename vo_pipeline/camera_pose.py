import cv2 as cv
import numpy as np
from exercise_helpers.decompose_essential_matrix import decomposeEssentialMatrix
from exercise_helpers.disambiguate_relative_pose import disambiguateRelativePose


def get_fundamental_matrix(keypoints_a, keypoints_b):
    fundamental_mat, mask = cv.findFundamentalMat(
        keypoints_a, keypoints_b, cv.FM_RANSAC, 3, 0.99
    )
    return fundamental_mat, mask


def essential_matrix_from_fundamental_matrix(fundamental_mat, K):
    return K.T @ fundamental_mat @ K


if __name__ == "__main__":
    import os
    import sys
    import time

    from correspondence import correspondence
    from helpers import load_images
    from exercise_helpers.camera_3d import drawCamera
    from exercise_helpers.linear_triangulation import linearTriangulation

    imgs = load_images("data/kitti/05/image_0/", start=0, end=2)
    K = np.array(
        [
            [7.188560000000e02, 0, 6.071928000000e02],
            [0, 7.188560000000e02, 1.852157000000e02],
            [0, 0, 1],
        ]
    )
    keypoints_a, keypoints_b = correspondence(imgs)

    t1 = time.time()
    fundamental_mat, _ = get_fundamental_matrix(keypoints_a, keypoints_b)
    essential_mat = essential_matrix_from_fundamental_matrix(fundamental_mat, K)
    Rots, u3 = decomposeEssentialMatrix(essential_mat)
    p1 = np.hstack([keypoints_a, np.ones((keypoints_a.shape[0], 1))]).T
    p2 = np.hstack([keypoints_b, np.ones((keypoints_b.shape[0], 1))]).T
    t2 = time.time()
    R_C2_W,T_C2_W = disambiguateRelativePose(Rots, u3, p1, p2, K, K)
    t3 = time.time()

    print(f"Time to calculate without disambiguateRelativePose: {t2-t1}")
    print(f"Time to calculate disambiguateRelativePose: {t3-t2}")
    print(f"Roataion: {R_C2_W}")
    print(f"Translation: {T_C2_W}")

    # plot keypoints from top view and camera
    import matplotlib.pyplot as plt
    from exercise_helpers.arrow_3d import Arrow3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    M1 = K @ np.eye(3,4)
    M2 = K @ np.c_[R_C2_W, T_C2_W]
    P = linearTriangulation(p1, p2, M1, M2)
    print(P.shape)  

    ax.scatter(P[0,:], P[1,:], P[2,:], marker="o", s=4)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    # Display camera pose
    drawCamera(ax, np.zeros((3,)), np.eye(3), length_scale = 2)
    ax.text(-0.1,-0.1,-0.1,"Cam 1")

    center_cam2_W = -R_C2_W.T @ T_C2_W
    drawCamera(ax, center_cam2_W, R_C2_W.T, length_scale = 2)
    ax.text(center_cam2_W[0]-0.1, center_cam2_W[1]-0.1, center_cam2_W[2]-0.1,'Cam 2')

    plt.show()




