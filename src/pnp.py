import cv2 as cv
import numpy as np
from state import FrameSate
import utils.geometry as geom
from klt import klt

def pnp(state_i: FrameSate, state_j: FrameSate, K: np.ndarray):
    matched_landmarks = state_i.landmarks.T
    pts_j, mask_klt = klt(state_i.keypoints.T, cv.imread(state_i.img_path, cv.IMREAD_GRAYSCALE), cv.imread(state_j.img_path, cv.IMREAD_GRAYSCALE))
    matched_keypoints = pts_j.squeeze()[mask_klt]
    matched_landmarks = matched_landmarks[mask_klt]

    success, rotation_vector_world_camj, translation_world_camj, mask_ransac = cv.solvePnPRansac(
        objectPoints=matched_landmarks,
        imagePoints=matched_keypoints, 
        cameraMatrix=K,
        distCoeffs=np.zeros((4, 1)),
        iterationsCount=5000,
        reprojectionError=2.0,
        confidence=0.99
    )
    
    if not success:
        # TODO: Fix WOKO dataset: happened only with WOKO dataset, not sure why though
        raise Exception("FUCK, mask is None! Something is wrong with the inputs.")

    rotation_world_camj, jacobian_world_camj = cv.Rodrigues(rotation_vector_world_camj)
    M_world_camj = np.vstack(
        [
            np.hstack([rotation_world_camj, translation_world_camj]),
            np.array([[0, 0, 0, 1]])
        ]
    )

    state_j.cam_to_world = np.linalg.inv(M_world_camj)
    state_j.keypoints = matched_keypoints[mask_ransac, :].squeeze().T
    state_j.landmarks = matched_landmarks[mask_ransac, :].squeeze().T
    return state_i, state_j


if __name__ == "__main__":
    from utils.dataloader import DataLoader, Dataset
    from initialization import initialize
    from plot_points_cameras import plot_points_cameras
    from twoDtwoD import FeatureDetector

    # select dataset
    dataset = Dataset.KITTI
    
    # load data
    # if you remove steps then all the images are used
    loader = DataLoader(dataset, start=0, stride=1, steps=100)
    print("Loading data...")

    loader.load_data()
    print("Data loaded!")

    K, poses, states = loader.get_data()
    print("Data retrieved!")

    # computes landmarks and pose for the first two frames
    if dataset == Dataset.KITTI:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]
    elif dataset == Dataset.PARKING:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]
    elif dataset == Dataset.MALAGA:
        initialize(states[0], states[4], K)
        init_states = [states[0], states[4]]
    elif dataset == Dataset.WOKO:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]
    
    steps = 0
    while steps < len(states) - 3:
        if steps == 90:
            break
        
        if states[steps].landmarks.shape[1] / states[0].landmarks.shape[1] < 0.2:
            initialize(states[steps], states[steps + 3], K)

        pnp(states[steps], states[steps+1], K)
        steps += 1

    plot_points_cameras(
        [state.landmarks for state in states[0:steps-1]],
        [state.cam_to_world for state in states[0:steps-1]]
    )
    