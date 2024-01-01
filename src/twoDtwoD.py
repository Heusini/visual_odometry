from enum import Enum

import cv2 as cv
import numpy as np

import utils.geometry as geom
from features import detect_features, match_features
from klt import klt
from plot_points_cameras import plot_points_cameras
from state import FrameState
from utils.decompose_essential_matrix import decomposeEssentialMatrix
from utils.disambiguate_relative_pose import disambiguateRelativePose
from utils.linear_triangulation import linearTriangulation
from utils.path_loader import PathLoader


class FeatureDetector(Enum):
    KLT = 0
    SIFT = 1


# Note that order is scale rotate translate
# When composing matrices this means T * R * S
def twoDtwoD(
    state_i: FrameState,
    state_j: FrameState,
    K: np.ndarray,  # camera intrinsics
    feature_detector: FeatureDetector = FeatureDetector.KLT,
):
    img_i = cv.imread(state_i.img_path, cv.IMREAD_GRAYSCALE)
    img_j = cv.imread(state_j.img_path, cv.IMREAD_GRAYSCALE)
    state_i.features = detect_features(img_i)
    pts_i = np.array([kp.pt for kp in state_i.features.keypoints])

    if feature_detector == FeatureDetector.KLT:
        # perform klt
        pts_j, mask = klt(pts_i, img_i, img_j)

        # select only good keypoints
        pos_i = pts_i[mask]
        pos_j = pts_j.squeeze()[mask]
    else:
        state_j.features = detect_features(img_j)
        mf_i, mf_j, _ = match_features(state_i.features, state_j.features)
        state_i.features = mf_i
        state_j.features = mf_j

        pos_i = mf_i.get_positions()
        pos_j = mf_j.get_positions()
        print(f"{len(pos_i)} keypoints matched")
        print(f"{len(pos_j)} keypoints matched")

    F, mask_f = geom.calc_fundamental_mat(pos_i, pos_j)
    E = geom.calc_essential_mat_from_fundamental_mat(F, K)
    p_i = np.hstack([pos_i, np.ones((pos_i.shape[0], 1))]).T
    p_j = np.hstack([pos_j, np.ones((pos_j.shape[0], 1))]).T

    p_i = p_i[:, mask_f.ravel() == 1]
    p_j = p_j[:, mask_f.ravel() == 1]
    R, T = decomposeEssentialMatrix(E)
    # Rotate -> Translate order
    R_cami_to_camj, T_cami_to_camj = disambiguateRelativePose(R, T, p_i, p_j, K, K)

    M_cami_to_camj = np.eye(4)
    M_cami_to_camj[:3, :] = np.c_[R_cami_to_camj, T_cami_to_camj]

    M_camj_to_cami = np.linalg.inv(M_cami_to_camj)
    M_cami_to_world = state_i.cam_to_world
    M_camj_to_world = M_cami_to_world @ M_camj_to_cami

    P_cami = linearTriangulation(p_i, p_j, K @ np.eye(3, 4), K @ M_cami_to_camj[0:3, :])

    # filer points behind camera and far away
    max_distance = 100
    mask = np.logical_and(
        P_cami[2, :] > 0, np.abs(np.linalg.norm(P_cami, axis=0)) < max_distance
    )
    P_cami = P_cami[:, mask]
    p_i = p_i[:, mask]
    p_j = p_j[:, mask]

    P_world = M_cami_to_world @ P_cami

    state_j.cam_to_world = M_camj_to_world
    state_j.landmarks = P_world[:3, :]
    state_i.landmarks = P_world[:3, :]

    # updating the keypoints in each state since there have been multiple
    # filtering steps that reduced the number of initial keypoints
    state_i.keypoints = p_i[:2, :]
    state_j.keypoints = p_j[:2, :]

    return state_i, state_j


def calculate_relative_pose(points_i, points_j, K):
    F, mask_f = geom.calc_fundamental_mat(points_i, points_j)
    E = geom.calc_essential_mat_from_fundamental_mat(F, K)
    p_i = np.hstack([points_i, np.ones((points_i.shape[0], 1))]).T
    p_j = np.hstack([points_j, np.ones((points_j.shape[0], 1))]).T

    p_i = p_i[:, mask_f.ravel() == 1]
    p_j = p_j[:, mask_f.ravel() == 1]
    R, T = decomposeEssentialMatrix(E)
    # Rotate -> Translate order
    R_cami_to_camj, T_cami_to_camj = disambiguateRelativePose(R, T, p_i, p_j, K, K)

    return R_cami_to_camj, T_cami_to_camj


def initialize_camera_poses(points_i, points_j, K: np.ndarray):
    R_cami_to_camj, T_cami_to_camj = calculate_relative_pose(points_i, points_j, K)
    M_cami_to_camj = np.eye(4)
    M_cami_to_camj[:3, :] = np.c_[R_cami_to_camj, T_cami_to_camj]

    M_camj_to_cami = np.linalg.inv(M_cami_to_camj)
    M_cami_to_world = np.eye(3, 4)
    M_camj_to_world = M_cami_to_world @ M_camj_to_cami

    p_i = np.hstack([points_i, np.ones((points_i.shape[0], 1))]).T
    p_j = np.hstack([points_j, np.ones((points_j.shape[0], 1))]).T
    P_cami = linearTriangulation(p_i, p_j, K @ np.eye(3, 4), K @ M_cami_to_camj[0:3, :])

    #######
    # opencv triangulatePoints
    # P_cami = cv.triangulatePoints(
    #     K @ np.eye(3, 4), K @ M_cami_to_camj[0:3, :], p_i[:2, :], p_j[:2, :]
    # )
    # print(P_cami.shape)
    #######

    # filer points behind camera and far away
    max_distance = 100
    mask = np.logical_and(
        P_cami[2, :] > 0, np.abs(np.linalg.norm(P_cami, axis=0)) < max_distance
    )
    P_cami = P_cami[:, mask]
    p_i = p_i[:, mask]
    p_j = p_j[:, mask]

    P_world = M_cami_to_world @ P_cami

    cam_to_world = M_camj_to_world
    landmarks = P_world[:3, :]

    # updating the keypoints in each state since there have been multiple
    # filtering steps that reduced the number of initial keypoints

    return cam_to_world, landmarks, mask


class DataSetEnum(Enum):
    KITTI = "kitti"
    PARKING = "parking"
    MALAGA = "malaga"


if __name__ == "__main__":
    steps = 30
    stride = 3
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

    path_loader = PathLoader(path, start=90, stride=stride)
    path_iter = iter(path_loader)

    states = []
    for i in range(steps):
        img_path = next(path_iter)
        state = FrameState(i, img_path)
        states.append(state)

    for i in range(steps - 1):
        twoDtwoD(states[i], states[i + 1], K)

    plot_points_cameras(
        [state.landmarks for state in states[0 : steps - 1]],
        [state.cam_to_world for state in states],
    )
