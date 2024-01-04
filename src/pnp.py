import cv2 as cv
import numpy as np

import utils.geometry as geom
from features import detect_features, match_features
from klt import klt
from state import FrameState
from twoDtwoD import twoDtwoD
from features import FeatureDetector


def pnp(points_2d: np.ndarray, points_3d: np.ndarray, K: np.ndarray, reprojection_error: float = 1.0, confidence: float = 0.9999, num_iterations: int = 50000):
    success, rotation_vector_world_camj, translation_world_camj, mask_ransac = (
        cv.solvePnPRansac(
            objectPoints=points_3d,
            imagePoints=points_2d,
            cameraMatrix=K,
            distCoeffs=np.zeros((4, 1)),
            iterationsCount=num_iterations,
            reprojectionError=reprojection_error,
            confidence=confidence,
        )
    )

    if not success:
        print("OpenCV PNP failed")
        Exception("PNP Failed")

    rotation_world_camj, jacobian_world_camj = cv.Rodrigues(rotation_vector_world_camj)
    M_world_camj = np.vstack(
        [
            np.hstack([rotation_world_camj, translation_world_camj]),
            np.array([[0, 0, 0, 1]]),
        ]
    )

    # evlt return mask_ransac here
    return np.linalg.inv(M_world_camj), mask_ransac


def pnp_old(state_i: FrameState, state_j: FrameState, K: np.ndarray):
    matched_landmarks = state_i.landmarks.T
    pts_j, mask_klt = klt(
        state_i.keypoints.T,
        cv.imread(state_i.img_path, cv.IMREAD_GRAYSCALE),
        cv.imread(state_j.img_path, cv.IMREAD_GRAYSCALE),
    )
    mask_klt = np.where(mask_klt.reshape(-1, 1) == True)[0]
    matched_keypoints = pts_j.squeeze()[mask_klt]
    matched_landmarks = matched_landmarks[mask_klt]

    success, rotation_vector_world_camj, translation_world_camj, mask_ransac = (
        cv.solvePnPRansac(
            objectPoints=matched_landmarks,
            imagePoints=matched_keypoints,
            cameraMatrix=K,
            distCoeffs=np.zeros((4, 1)),
            iterationsCount=10000,
            reprojectionError=1.0,
            confidence=0.9999,
        )
    )

    if not success:
        # TODO: Fix WOKO dataset: happened only with WOKO dataset, not sure why though
        raise Exception(
            "There are less than 8 points given in to the solve PNP method."
        )

    rotation_world_camj, jacobian_world_camj = cv.Rodrigues(rotation_vector_world_camj)
    M_world_camj = np.vstack(
        [
            np.hstack([rotation_world_camj, translation_world_camj]),
            np.array([[0, 0, 0, 1]]),
        ]
    )

    state_j.cam_to_world = np.linalg.inv(M_world_camj)
    state_j.keypoints = matched_keypoints[mask_ransac, :].squeeze().T
    state_j.landmarks = matched_landmarks[mask_ransac, :].squeeze().T

    print("{} Landmarks in frame {}.".format(state_i.landmarks.shape[1], state_i.t))
    print("{} Landmarks in frame {}.".format(state_j.keypoints.shape[1], state_j.t))
    return state_j


if __name__ == "__main__":
    from initialization import initialize
    from plot_points_cameras import plot_points_cameras
    from twoDtwoD import FeatureDetector
    from utils.dataloader import DataLoader, Dataset

    # select dataset
    dataset = Dataset.PARKING

    # load data
    # if you remove steps then all the images are used
    loader = DataLoader(dataset, start=0, stride=1, steps=100)
    print("Loading data...")

    loader.load_data()
    print("Data loaded!")

    K, poses, states = loader.get_data()
    print("Data retrieved!")

    start_frame = 3
    feature_detector = FeatureDetector.KLT
    features_detector_init = FeatureDetector.SIFT
    # computes landmarks and pose for the first two frames
    states[0].features = detect_features(
        cv.imread(states[0].img_path, cv.IMREAD_GRAYSCALE)
    )
    if dataset == Dataset.KITTI:
        initialize(states[0], states[start_frame], K, features_detector_init)
        init_states = [states[0], states[3]]
    elif dataset == Dataset.PARKING:
        initialize(states[0], states[start_frame], K, features_detector_init)
        init_states = [states[0], states[3]]
    elif dataset == Dataset.MALAGA:
        initialize(states[0], states[start_frame], K, features_detector_init)
        init_states = [states[0], states[4]]
    elif dataset == Dataset.WOKO:
        initialize(states[0], states[start_frame], K, features_detector_init)
        init_states = [states[0], states[3]]

    steps = 0
    run_for_frames = 99
    for steps in range(run_for_frames):
        print(steps)
        state_i = states[steps]
        state_j = states[steps + 1]

        img1 = cv.imread(state_i.img_path, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(state_j.img_path, cv.IMREAD_GRAYSCALE)

        if steps % 10 == 0 and steps != 0:
            cam_to_world, landmarks, keypoints = twoDtwoD(
                states[steps], states[steps + 5], K, FeatureDetector.SIFT
            )

            states[steps + 5].cam_to_world = cam_to_world
            states[steps + 5].landmarks = landmarks
            states[steps + 5].keypoints = keypoints

        matched_landmarks = state_i.landmarks.T
        pts, mask = klt(state_i.keypoints.T, img1, img2)

        mask_klt = np.where(mask.reshape(-1, 1) == True)[0]
        matched_keypoints = pts.squeeze()[mask_klt]
        matched_landmarks = matched_landmarks[mask_klt]

        print(f"{matched_keypoints.shape} keypoints lol matched")
        print(f"{matched_landmarks.shape} landmarks lol matched")

        camera_pose, mask_ransac = pnp(matched_keypoints, matched_landmarks, K)

        state_j.keypoints = matched_keypoints[mask_ransac, :].squeeze().T
        state_j.landmarks = matched_landmarks[mask_ransac, :].squeeze().T
        state_j.cam_to_world = camera_pose
        # print("{} Landmarks in frame {}.".format(state_i.landmarks.shape[1], state_i.t))
        # print("{} Landmarks in frame {}.".format(state_j.keypoints.shape[1], state_j.t))
        steps += 1

    plot_points_cameras(
        [state.landmarks for state in states[0:steps]],
        [state.cam_to_world for state in states[0:steps]],
    )
