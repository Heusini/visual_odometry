import cv2 as cv
import numpy as np

from features import Features, detect_features, match_features
from plot_points_cameras import plot_points_cameras
from pnp import pnp
from twoDtwoD import initialize_camera_poses, twoDtwoD

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

    start_frame = 2
    img1 = cv.imread(states[0].img_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(states[start_frame].img_path, cv.IMREAD_GRAYSCALE)

    features1 = detect_features(img1)
    features2 = detect_features(img2)

    mf_i, mf_j, _ = match_features(features1, features2)
    pos_i = mf_i.get_positions()
    pos_j = mf_j.get_positions()
    print(f"{len(pos_i)} keypoints matched")
    print(f"{len(pos_j)} keypoints matched")

    camj_to_world, landmarks, mask = initialize_camera_poses(pos_i, pos_j, K)

    states[0].landmarks = landmarks.T
    states[0].features = Features(mf_i.keypoints[mask], mf_i.descriptors[mask])
    states[start_frame].features = Features(
        mf_j.keypoints[mask], mf_j.descriptors[mask]
    )
    states[start_frame].landmarks = landmarks.T
    states[start_frame].cam_to_world = camj_to_world

    state0 = states[start_frame]
    run_steps = 10
    camera_poses = [np.eye(4), state0.cam_to_world]
    for step in range(start_frame + 1, min(run_steps, len(states))):
        state_j = states[step]
        imgj = cv.imread(state_j.img_path, cv.IMREAD_GRAYSCALE)

        featuresj = detect_features(imgj)

        mf_i, mf_j, mask = match_features(state0.features, featuresj)

        print(state0.landmarks[mask, :].shape)
        print(mf_j.get_positions().shape)
        camera_pose, mask_ransac = pnp(
            mf_j.get_positions(), state0.landmarks[mask, :], K
        )
        camera_poses.append(camera_pose)
    plot_points_cameras([state0.landmarks.T], camera_poses)
