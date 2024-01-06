import numpy as np
import cv2 as cv

def disambiguateRelativePose(Rots, u3, points0_h, points1_h, K1, K2):
    """DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among four
    possible configurations) by returning the one that yields points lying in front of
    the image plane (with positive depth).

    Arguments:
      Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
      u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
      p1   -  3xN homogeneous coordinates of point correspondences in image 1
      p2   -  3xN homogeneous coordinates of point correspondences in image 2
      K1   -  3x3 calibration matrix for camera 1
      K2   -  3x3 calibration matrix for camera 2

    Returns:
      R -  3x3 the correct rotation matrix
      T -  3x1 the correct translation vector
      P_cami - Nx4 dehomogenized 3D pointcloud 

      where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
      from the world coordinate system (identical to the coordinate system of camera 1)
      to camera 2.
    """
    pass

    # Projection matrix of camera 1
    M1 = K1 @ np.eye(3, 4)

    total_points_in_front_best = -np.inf
    for iRot in range(2):
        R_C2_C1_test = Rots[:, :, iRot]
        for iSignT in range(2):
            T_C2_C1_test = u3 * (-1) ** iSignT

            # projection matrix camera 2
            M2 = K2 @ np.c_[R_C2_C1_test, T_C2_C1_test]

            # 3D pointcloud w.r.t. camera 1
            P_C1 = cv.triangulatePoints(M1, M2, points0_h[:2, :], points1_h[:2, :])
            P_C1 /= P_C1[3, :]

            # project in both cameras
            P_C2 = np.c_[R_C2_C1_test, T_C2_C1_test] @ P_C1

            # Number of points in front of both cameras
            total_points_in_front = np.sum(P_C1[2, :] > 0) + np.sum(P_C2[2, :] > 0)

            if total_points_in_front > total_points_in_front_best:
                # Keep the rotation that gives the highest number of points
                # in front of both cameras
                R = R_C2_C1_test
                T = T_C2_C1_test
                P_cami = P_C1
                total_points_in_front_best = total_points_in_front
    
    # obtain homogenous transformation matrix
    T_cami_camj = np.eye(4)
    T_cami_camj[:3, :3] = R
    T_cami_camj[:3, 3] = T.ravel()

    # Remove points behind the camera and far away
    max_point_distance = 100
    min_depth_distance = min(0, T_cami_camj[2, 2])
    min_depth_distance = 0
    close_points = np.linalg.norm(P_cami[:3, :], axis=0) <= max_point_distance
    not_behind_camera = P_cami[2, :] > min_depth_distance

    mask = np.logical_and(close_points, not_behind_camera)

    P_cami = P_cami[:, mask].T

    print(f"2D2D/DISAMBIGUATE: {np.sum(~mask)} rejected points during 3D reconstruction")
    return T_cami_camj, P_cami, mask
