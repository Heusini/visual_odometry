import cv2 as cv
import numpy as np
from camera_pose import Camera, essential_matrix_from_fundamental_matrix, get_fundamental_matrix
from exercise_helpers.decompose_essential_matrix import decomposeEssentialMatrix
from exercise_helpers.disambiguate_relative_pose import disambiguateRelativePose
from exercise_helpers.normalise_2D_pts import normalise2DPts
from exercise_helpers.linear_triangulation import linearTriangulation

def sfm(
    keypoints_a: np.ndarray,
    keypoints_b: np.ndarray,
    K : np.ndarray,
) -> (np.ndarray, Camera, Camera):
    """
    Associated task:
        3.3. Estimate the relative pose between the frames and triangulate a point cloud of 3D landmarks
        (exercise 6)

    Parameters
    ----------
    keypoints_a : np.ndarray
        Shape is (K, 2) where K is the number of keypoints.
    keypoints_b : np.ndarray
        Shape is (K, 2) where K is the number of keypoints.

    Returns ( P , (R, T), (K_a, K_b) )
    -------
    P : np.ndarray
        3D Keypoints

    C_a : Camera
        Camera object of camera a

    C_b : Camera
        Camera object of camera b
    """

    p1 = np.hstack([keypoints_a, np.ones((keypoints_a.shape[0], 1))]).T
    p2 = np.hstack([keypoints_b, np.ones((keypoints_b.shape[0], 1))]).T

    F, _ = get_fundamental_matrix(p1[:2, :].T, p2[:2, :].T)
    E = essential_matrix_from_fundamental_matrix(F, K)

    Rots, u3 = decomposeEssentialMatrix(E)
    R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, p1, p2, K, K)

    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R_C2_W, T_C2_W]

    P = linearTriangulation(p1, p2, M1, M2)

    T_W_C2 = -R_C2_W.T @ T_C2_W
    T_C2_W = T_C2_W.reshape((3, 1))

    cam_a = Camera(np.eye(3, 3), np.zeros((3, 1)), K, "Cam 1")
    cam_b = Camera(R_C2_W, T_C2_W, K, "Cam 2")

    return P, cam_a, cam_b
