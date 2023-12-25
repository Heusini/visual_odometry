import cv2 as cv
import numpy as np
from enum import Enum
from state import FrameSate, Transform3D
import utils.geometry as geom

from features import detect_features, match_features

from utils.decompose_essential_matrix import decomposeEssentialMatrix
from utils.disambiguate_relative_pose import disambiguateRelativePose
from utils.linear_triangulation import linearTriangulation
from utils.path_loader import PathLoader
from plot_points_cameras import plot_points_cameras

from klt import klt

# Note that order is scale rotate translate
# When composing matrices this means T * R * S
def pnp(
    state_i : FrameSate,
    state_j : FrameSate,
    K : np.ndarray, # camera intrinsics
) :
    img_i = cv.imread(state_i.img_path, cv.IMREAD_GRAYSCALE)
    img_j = cv.imread(state_j.img_path, cv.IMREAD_GRAYSCALE)
    pts_i = np.array([
        kp.pt for kp in state_i.features.keypoints
    ])

    # perform klt
    pts_j, mask = klt(pts_i, img_i, img_j)

    # select only good keypoints
    pos_mf_i = pts_i[mask]
    pos_mf_j = pts_j.squeeze()[mask]

    F, mask_f = geom.calc_fundamental_mat(pos_mf_i, pos_mf_j)
    E = geom.calc_essential_mat_from_fundamental_mat(F, K)
    p_i = np.hstack([pos_mf_i, np.ones((pos_mf_i.shape[0], 1))]).T
    p_j = np.hstack([pos_mf_j, np.ones((pos_mf_j.shape[0], 1))]).T

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

    P_cami = linearTriangulation(
        p_i, 
        p_j, 
        K @ np.eye(3, 4),
        K @ M_cami_to_camj[0:3, :])

    #filer points behind camera and far away
    max_distance = 100
    mask = np.logical_and(P_cami[2, :] > 0, np.abs(np.linalg.norm(P_cami, axis=0)) < max_distance)
    P_cami = P_cami[:, mask]

    P_world = M_cami_to_world @ P_cami

    state_j.cam_to_world = M_camj_to_world
    state_j.landmarks = P_world
    state_i.landmarks = P_world

    return state_i, state_j

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
        state = FrameSate(i, img_path)
        states.append(state)

    for i in range(steps-1):
        pnp(states[i], states[i+1], K)
    
    plot_points_cameras(
        [state.landmarks for state in states[0:steps-1]],
        [state.cam_to_world for state in states]
    )