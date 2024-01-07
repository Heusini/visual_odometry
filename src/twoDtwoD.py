from enum import Enum

import cv2 as cv
import numpy as np

import utils.geometry as geom
from features import detect_features, match_features, matching_klt
from plot_points_cameras import plot_points_cameras
from state import FrameState
from utils.decompose_essential_matrix import decomposeEssentialMatrix
from utils.disambiguate_relative_pose import disambiguateRelativePose
from features import FeatureDetector
from utils.utils import hom_inv

# Note that order is scale rotate translate
# When composing matrices this means T * R * S
def twoDtwoD(
    state_i: FrameState,
    state_j: FrameState,
    img_paths: list,
    K: np.ndarray,  # camera intrinsics
    feature_detector: FeatureDetector = FeatureDetector.KLT,
    no_feature_detection: bool = False,
    pos_i: np.ndarray = None,
    pos_j: np.ndarray = None
):
    # convert img to grayscale
    img_i = cv.imread(state_i.img_path, cv.IMREAD_GRAYSCALE)
    img_j = cv.imread(state_j.img_path, cv.IMREAD_GRAYSCALE)

    if not no_feature_detection:
        if feature_detector == FeatureDetector.KLT.value:
            # Extract feature (SIFT) in image i and tracked them to image
            # j using KLT
            state_i.features = detect_features(img_i)
            pts_i = state_i.features.get_positions()
            
            # perform klt
            pts_j, mask = matching_klt(img_paths, pts_i)
            
            # select only good keypoints
            pos_i = pts_i[mask, :]
            pos_j = pts_j[mask, :]
        else:
            # Extract features in both image i and image j and match them using KNN
            state_i.features = detect_features(img_i)
            state_j.features = detect_features(img_j)

            # matche SIFT features and returned best matches
            mf_i, mf_j = match_features(state_i.features, state_j.features)
            state_i.features = mf_i
            state_j.features = mf_j

            # get the position in pixels of each feature
            pos_i = mf_i.get_positions()
            pos_j = mf_j.get_positions()

    # Calculate the Fundamental matrix using 8 point algorithm with RANSAC
    F, mask = geom.calc_fundamental_mat(pos_i, pos_j)
    
    ## Select only inliers points
    inliers = mask.ravel() == 1
    pos_i = pos_i[inliers, :]
    pos_j = pos_j[inliers, :]

    # get essential matrix
    E = geom.calc_essential_mat_from_fundamental_mat(F, K)

    # Convert keypoints to homogenous coordinates
    p_i = np.hstack([pos_i, np.ones((pos_i.shape[0], 1))])
    p_j = np.hstack([pos_j, np.ones((pos_j.shape[0], 1))])

    # Decompose the essential matrix into R and T
    R, T = decomposeEssentialMatrix(E)
    
    # Disambiguate between different translations and rotations, where the solutions is the one with the 
    # most points in front of both camera
    T_cami_to_camj, P_cami, mask_reconstruction = disambiguateRelativePose(R, T, p_i.T, p_j.T, K, K)
    
    # use only feasible keypoints
    p_i = p_i[mask_reconstruction]
    p_j = p_j[mask_reconstruction]

    T_camj_to_world = hom_inv(T_cami_to_camj) @ state_i.cam_to_world
    P_world = (hom_inv(state_i.cam_to_world) @ P_cami.T).T

    return T_camj_to_world, P_world[:, :3], p_j[:, :2], mask_reconstruction, inliers

def calculate_relative_pose(points_i, points_j, K):
    F, mask_f = geom.calc_fundamental_mat(points_i, points_j)
    E = geom.calc_essential_mat_from_fundamental_mat(F, K)
    p_i = np.hstack([points_i, np.ones((points_i.shape[0], 1))]).T
    p_j = np.hstack([points_j, np.ones((points_j.shape[0], 1))]).T

    p_i = p_i[:, mask_f.ravel() == 1]
    p_j = p_j[:, mask_f.ravel() == 1]
    R, T = decomposeEssentialMatrix(E)
    # Rotate -> Translate order
    T, _, _ = disambiguateRelativePose(R, T, p_i, p_j, K, K)
    R_cami_to_camj = T[:3, :3]
    T_cami_to_camj = T[:3, 3]
    return R_cami_to_camj, T_cami_to_camj

# TODO: kinda confusing that the returned landmarks are already masked
def initialize_camera_poses(points_i, points_j, K: np.ndarray):
    R_cami_to_camj, T_cami_to_camj = calculate_relative_pose(points_i, points_j, K)
    M_cami_to_camj = np.eye(4)
    M_cami_to_camj[:3, :] = np.c_[R_cami_to_camj, T_cami_to_camj]

    M_camj_to_cami = np.linalg.inv(M_cami_to_camj)
    M_cami_to_world = np.eye(3, 4)
    M_camj_to_world = M_cami_to_world @ M_camj_to_cami

    p_i = np.hstack([points_i, np.ones((points_i.shape[0], 1))]).T
    p_j = np.hstack([points_j, np.ones((points_j.shape[0], 1))]).T

    # opencv triangulatePoints
    P_cami = cv.triangulatePoints(
        K @ np.eye(3, 4), K @ M_cami_to_camj[0:3, :], p_i[:2, :], p_j[:2, :]
    )
    P_cami /= P_cami[3, :]

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

from utils.dataloader import DataLoader, Dataset
if __name__ == "__main__":
      # select dataset
    dataset = Dataset.PARKING
    
    steps = 20
    stride = 4
    # load data
    # if you remove steps then all the images are used
    loader = DataLoader(dataset, start=105, stride=stride, steps=steps)
    print("Loading data...")

    loader.load_data()
    print("Data loaded!")

    K, poses, states = loader.get_data()
    for i in range(steps - 1):
        cam_to_world, landmarks, keypoints = twoDtwoD(
            states[i], states[i + 1], K, feature_detector=FeatureDetector.SIFT
        )

        states[i + 1].cam_to_world = cam_to_world
        states[i + 1].landmarks = landmarks
        states[i + 1].keypoints = keypoints

    plot_points_cameras(
        [state.landmarks for state in states[0 : steps - 1]],
        [state.cam_to_world for state in states],
    )
