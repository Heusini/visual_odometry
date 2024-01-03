import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from dashboard import create_default_dashboard, cv_scatter
from features import Features, detect_features, match_features, detect_features_shi_tomasi, matching_klt
from plot_points_cameras import plot_points_cameras
from pnp import pnp
from twoDtwoD import initialize_camera_poses, twoDtwoD


def update_plot(img, points_2d, points_3d, camera_poses, refined_camera_poses, ax1, ax2, ax3):
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
    ax2.scatter(x_cams, y_cams, s=20, color='red')

    x_cams = []
    y_cams = []
    for cam in refined_camera_poses:
        centerpoint = cam @ np.array([0, 0, 0, 1])
        x_cams.append(centerpoint[0])
        y_cams.append(centerpoint[2])

    x_min = np.min(x_cams)
    x_max = np.max(x_cams)
    y_min = np.min(y_cams)
    y_max = np.max(y_cams)
    ax2.set_xlim(x_min - 0.5, x_max + 0.5)
    ax2.set_ylim(y_min - 0.5, y_max + 0.5)
    ax2.scatter(x_cams, y_cams, s=20, color='green', marker='x')

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
    from klt import klt
    from utils.non_linear_refinement import refine_camera_pose

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

    start_frame = 13
    img1 = cv.imread(states[0].img_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(states[start_frame].img_path, cv.IMREAD_GRAYSCALE)

    keypoints_0 = detect_features_shi_tomasi(img1, max_num=1000, threshold=0.03, non_maxima_suppression_size=10)
    keypoints_start_Frame, mask_good = matching_klt(states[:start_frame+1], keypoints_0, dict(winSize=(31, 31), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 6)))
    
    states[0].keypoints = keypoints_0[mask_good, :]
    states[start_frame].keypoints = keypoints_start_Frame.squeeze()[mask_good]

    for i in keypoints_0:
        x,y = i.ravel()
        cv.circle(img1,(int(x),int(y)),3,255,-1)
    
    for i in keypoints_start_Frame:
        x,y = i.ravel()
        cv.circle(img2,(int(x),int(y)),3,255,-1)

    visualize_points_j = states[start_frame].keypoints
    visualize_points_j[:, 0] += img1.shape[1]
    img = np.hstack([img1, img2])
    for pt1, pt2 in zip(states[0].keypoints, visualize_points_j):
        cv.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), 0, thickness=1)
    plt.imshow(img),plt.show()

    camj_to_world, landmarks, mask = initialize_camera_poses(states[0].keypoints, states[start_frame].keypoints, K)

    states[0].landmarks = landmarks.T
    states[0].keypoints = states[0].keypoints[mask]

    states[start_frame].keypoints = states[start_frame].keypoints[mask]
    states[start_frame].landmarks = landmarks.T
    states[start_frame].cam_to_world = camj_to_world

    state0 = states[start_frame]

    run_steps = 30
    camera_poses = [np.eye(4)]
    refined_camera_poses = [np.eye(4)]

    plt.ion()
    fig, ax = create_default_dashboard()
    ax1, ax2, ax3 = ax
    plt.show()

    for step in range(1, min(run_steps, len(states))):
        state_i = states[step - 1]
        state_j = states[step]
        imgj = cv.imread(state_j.img_path, cv.IMREAD_GRAYSCALE)

        landmarks = state_i.landmarks
        pts_j, mask_klt = klt(
            state_i.keypoints,
            cv.imread(state_i.img_path, cv.IMREAD_GRAYSCALE),
            cv.imread(state_j.img_path, cv.IMREAD_GRAYSCALE),
            dict(winSize=(31, 31), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 6))
        )
        mask_klt = np.where(mask_klt.reshape(-1, 1) == True)[0]
        matched_keypoints_i = state_i.keypoints[mask_klt, :]
        matched_keypoints_j = pts_j.squeeze()[mask_klt]
        matched_landmarks = landmarks[mask_klt, :]

        print(f"#Landmarks: {matched_landmarks.shape}")

        camera_pose, mask_ransac = pnp(
            matched_keypoints_j,
            matched_landmarks,
            K,
            reprojection_error=1.0,
            confidence=0.9999,
            num_iterations=5000 
        )

        matched_keypoints_j = matched_keypoints_j[mask_ransac.flatten(), :]
        matched_landmarks = matched_landmarks[mask_ransac.flatten(), :]

        state_j.landmarks = matched_landmarks
        state_j.keypoints = matched_keypoints_j
        
        refined_camera_pose = refine_camera_pose(keypoints=matched_keypoints_j, landmarks=matched_landmarks, C_guess=camera_pose, K=K)

        refined_camera_poses.append(refined_camera_pose)
        camera_poses.append(camera_pose)
        update_plot(
            imgj,
            [matched_keypoints_j.T, matched_keypoints_i.T],
            matched_landmarks,
            refined_camera_poses,
            camera_poses,
            ax1,
            ax2,
            ax3,
        )
        plt.draw()
        plt.waitforbuttonpress()