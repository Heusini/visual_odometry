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

def pnp(
    state_i : FrameSate,
    state_j : FrameSate,
    K : np.ndarray, # camera intrinsics
) :

    mf_i, mf_j = match_features(state_i.features, state_j.features)

    pos_mf_i = mf_i.get_positions()
    pos_mf_j = mf_j.get_positions()

    F, mask_f = geom.calc_fundamental_mat(pos_mf_i, pos_mf_j)
    E = geom.calc_essential_mat_from_fundamental_mat(F, K)
    p_i = np.hstack([pos_mf_i, np.ones((pos_mf_i.shape[0], 1))]).T
    p_j = np.hstack([pos_mf_j, np.ones((pos_mf_j.shape[0], 1))]).T

    p_i = p_i[:, mask_f.ravel() == 1]
    p_j = p_j[:, mask_f.ravel() == 1]
    R, T = decomposeEssentialMatrix(E)
    R_camj_to_cami, T_camj_to_cami = disambiguateRelativePose(R, T, p_i, p_j, K, K)

    #M1 = K @ state_i.world_to_cam.to_mat()[0:3, :]
    #M2 = K @ transform_j.to_mat()[0:3, :]
    # P_world = linearTriangulation(p_i, p_j, M1, M2)
    # # filer points behind camera and far away
    # max_distance = 100
    # P_cam_i = state_i.world_to_cam.to_mat() @ P_world
    # mask = np.logical_and(P_cam_i[2, :] > 0, np.abs(np.linalg.norm(P_cam_i, axis=0)) < max_distance)
    # P_world = P_world[:, mask]

    # M_to_cam = np.linalg.inv(state_i.world_to_cam.to_mat())
    # M_to_world = np.linalg.inv(M_to_cam)

    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R_camj_to_cami, T_camj_to_cami]
    P_cami = linearTriangulation(p_i, p_j, M1, M2)

    # filer points behind camera and far away
    max_distance = 100
    mask = np.logical_and(P_cami[2, :] > 0, np.abs(np.linalg.norm(P_cami, axis=0)) < max_distance)
    P_cami = P_cami[:, mask]

    R_camj_to_world = np.linalg.inv(R_camj_to_cami) @ state_i.cam_to_world.R
    T_camj_to_world = state_i.cam_to_world.t - np.reshape(T_camj_to_cami, (3, 1))

    # TODO: I have no idea why we need to Rotate and then Translate here, meanwhile when getting the camera pos we need to Translate and then Rotate
    P_world = state_i.cam_to_world.Rt() @ P_cami

    state_j.cam_to_world = Transform3D(R_camj_to_world, T_camj_to_world)
    state_j.landmarks = P_world
    state_i.landmarks = P_world

    return state_i, state_j

class DataSetEnum(Enum):
    KITTI = "kitti"
    PARKING = "parking"
    MALAGA = "malaga"

if __name__ == "__main__":
    
    steps = 10
    stride = 2
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

    path_loader = PathLoader(path, start=10, stride=stride)
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