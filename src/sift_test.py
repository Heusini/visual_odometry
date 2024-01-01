import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from dashboard import create_default_dashboard, cv_scatter
from features import Features, detect_features, match_features
from plot_points_cameras import plot_points_cameras
from pnp import pnp
from twoDtwoD import initialize_camera_poses, twoDtwoD


def update_plot(img, points_2d, points_3d, camera_poses, ax1, ax2, ax3):
    color = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for i in range(len(points_2d)):
        img = cv_scatter(img, points_2d[i], color[i], 3, -1)
    ax1.imshow(img)

    x_cams = []
    y_cams = []
    for cam in camera_poses:
        centerpoint = cam @ np.array([0, 0, 0, 1])
        x_cams.append(centerpoint[0])
        y_cams.append(centerpoint[2])

    x_min = np.min(x_cams)
    x_max = np.max(x_cams)
    y_min = np.min(y_cams)
    y_max = np.max(y_cams)
    ax2.set_xlim(x_min - 0.5, x_max + 0.5)
    ax2.set_ylim(y_min - 0.5, y_max + 0.5)
    ax2.scatter(x_cams, y_cams, s=20)

    x_min = np.min(points_3d[:, 0])
    x_max = np.max(points_3d[:, 0])
    y_min = np.min(points_3d[:, 2])
    y_max = np.max(points_3d[:, 2])
    ax3.set_xlim(x_min - 0.5, x_max + 0.5)
    ax3.set_ylim(y_min - 0.5, y_max + 0.5)
    print(points_3d.shape)
    ax3.scatter(points_3d[:, 0], points_3d[:, 2], s=20)
    ax3.title.set_text("matched landmarks")


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
    print(f"State0 Landmark shape: {state0.landmarks.shape}")
    print(f"State0 features: {state0.features.get_positions().shape}")
    run_steps = 10
    camera_poses = [np.eye(4), state0.cam_to_world]

    plt.ion()
    fig, ax1, ax2, ax3 = create_default_dashboard()
    plt.show()

    for step in range(start_frame + 1, min(run_steps, len(states))):
        state_j = states[step]
        imgj = cv.imread(state_j.img_path, cv.IMREAD_GRAYSCALE)

        featuresj = detect_features(imgj)
        print(featuresj.get_positions().shape)
        print(state0.features.get_positions().shape)

        mf_i, mf_j, mask = match_features(state0.features, featuresj, threshold=0.9)
        print(mask.shape)
        print(mf_i.get_positions().shape)
        print(mf_j.get_positions().shape)
        print(f"Landmark shape: {state0.landmarks.shape}")

        print(state0.landmarks[mask, :].shape)
        print(mf_j.get_positions().shape)
        camera_pose, mask_ransac = pnp(
            mf_j.get_positions(), state0.landmarks[mask, :], K
        )
        camera_poses.append(camera_pose)
        update_plot(
            imgj,
            [mf_j.get_positions().T, mf_i.get_positions().T],
            state0.landmarks[mask, :],
            camera_poses,
            ax1,
            ax2,
            ax3,
        )
        plt.draw()
        input("Press Enter to continue...")

    # plot_points_cameras([state0.landmarks.T], camera_poses)
