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
    R, T = decomposeEssentialMatrix(essential_mat)
    p1 = np.hstack([keypoints_a, np.ones((keypoints_a.shape[0], 1))]).T
    p2 = np.hstack([keypoints_b, np.ones((keypoints_b.shape[0], 1))]).T
    t2 = time.time()
    R, T = disambiguateRelativePose(R, T, p1, p2, K, K)
    t3 = time.time()

    print(f"Time to calculate without disambiguateRelativePose: {t2-t1}")
    print(f"Time to calculate disambiguateRelativePose: {t3-t2}")
    print(f"Roataion: {R}")
    print(f"Translation: {T}")
