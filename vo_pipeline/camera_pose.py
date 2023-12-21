import time
from enum import Enum
from typing import List

import cv2 as cv
import numpy as np
from camera import (
    Camera,
    essential_matrix_from_fundamental_matrix,
    get_fundamental_matrix,
)
from correspondence import correspondence
from exercise_helpers.decompose_essential_matrix import decomposeEssentialMatrix
from exercise_helpers.disambiguate_relative_pose import disambiguateRelativePose
from exercise_helpers.linear_triangulation import linearTriangulation
from helpers import draw_camera_wireframe


class DataSetEnum(Enum):
    KITTI = "kitti"
    PARKING = "parking"
    MALAGA = "malaga"

def plot_plotly(Ps : List[np.ndarray], cameras: List[Camera]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plotly_fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scatter"}, {"type": "scatter3d"}]]
    )
    plotly_fig.update_scenes(aspectmode="data")
    # draw 2D xz scatter plot plus camera position
    for P in Ps:
        scatter_2D = go.Scatter(
            x=P[0, :],
            y=P[2, :],
            mode="markers",
            marker=dict(
                size=3,
                colorscale="Viridis",  # choose a colorscale
                opacity=0.8,
            ),
        )
        plotly_fig.add_trace(scatter_2D, 1, 1)

    for camera in cameras:
        centerpoint = -camera.T_to_cam.ravel()
        scatter_2D = go.Scatter(
            x=[centerpoint[0]],
            y=[centerpoint[2]],
            mode="markers+text",
            textposition="bottom center",
            text=[f"Cam: {camera.id}"],
            marker=dict(size=5, colorscale="Viridis", opacity=0.8, symbol=4),
        )
        plotly_fig.add_trace(scatter_2D, 1, 1)


    # draw 3D scatter plot plus camera frames
    for P in Ps:
        scatter_3d = go.Scatter3d(
            x=P[0, :],
            y=P[1, :],
            z=P[2, :],
            mode="markers",
            marker=dict(
                size=3,
                # color="red",  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=0.8,
            ),
        )
        plotly_fig.add_trace(scatter_3d, 1, 2)

    colors = [
        "black",
        "green",
        "blue",
        "yellow",
        "orange",
        "purple",
        "pink",
        "brown",
    ]

    count = 0
    for camera in cameras:
        camera_wireframe = camera.draw_camera_wireframe(
            0.5, 0.5, f"Cam: {camera.id}", colors[count]
        )
        count = (count + 1) % len(colors)
        for line in camera_wireframe:
            plotly_fig.add_trace(line, 1, 2)

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=2, z=-2)
    )

    plotly_fig.update_layout(scene_camera=camera, title='default')
    plotly_fig.show()


def estimate_next_camera_pose(
    cam1: Camera, kp_1: np.ndarray, kp_2: np.ndarray
) -> ((np.ndarray, np.ndarray), np.ndarray):
    p1 = np.hstack([kp_1, np.ones((kp_1.shape[0], 1))]).T
    p2 = np.hstack([kp_2, np.ones((kp_2.shape[0], 1))]).T

    # mask_f contains the indices of all inlier points obtained from RANSAC
    F, mask_f = get_fundamental_matrix(p1[:2, :].T, p2[:2, :].T)
    E = essential_matrix_from_fundamental_matrix(F, K)

    # selecting only the inlier points (from RAMSAC mask)
    p1 = p1[:, mask_f.ravel() == 1]
    p2 = p2[:, mask_f.ravel() == 1]

    # camera1 in world coordinates
    T_W_C1 = np.reshape(cam1.translation, (3, 1))
    R_W_C1 = cam1.R_to_world

    # find rotation and translation from camera 2 to camera 1
    R_C2_C1, T_C2_C1 = decomposeEssentialMatrix(E)

    # make sure that the points are in front of the camera
    R_C2_C1, T_C2_C1 = disambiguateRelativePose(R_C2_C1, T_C2_C1, p1, p2, K, K)
    T_C2_C1 = T_C2_C1.reshape((3, 1))

    # compute the camera projection matrices
    M1 = K @ np.c_[cam1.R_to_world, cam1.translation]
    # position of camera2 in world coordinates
    T_W_C2 = T_W_C1 - R_C2_C1 @ T_C2_C1
    R_W_C2 = R_W_C1 @ R_C2_C1.T
    M2 = K @ np.c_[R_W_C2, T_W_C2]
    print(f"Camera matrix 1: {M1}")
    print(f"Camera matrix 2: {M2}")
    P_W = linearTriangulation(p1, p2, M1, M2)

    # project points into cam1 space
    P_C1 = M1 @ P_W
    mask = np.logical_and(P_C1[2, :] > 0, P_C1[2, :] < 100)
    P_W = P_W[:, mask]

    return (R_W_C2, T_W_C2), P_W


# test stuff
if __name__ == "__main__":
    imgs = []
    # imgs = load_images("data/kitti/05/image_0/", start=0, end=2)
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
        img1 = cv.imread("data/kitti/05/image_0/000000.png")
        img2 = cv.imread("data/kitti/05/image_0/000004.png")
        img3 = cv.imread("data/kitti/05/image_0/000008.png")
        K = K_kitty
    elif dataset == DataSetEnum.PARKING:
        img1 = cv.imread("data/parking/images/img_00000.png")
        img2 = cv.imread("data/parking/images/img_00003.png")
        img3 = cv.imread("data/parking/images/img_00006.png")
        K = K_parking
    elif dataset == DataSetEnum.MALAGA:
        pass

    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    img3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
    imgs = np.asarray(imgs)
    cam1 = Camera(1, K, img1)
    cam2 = Camera(2, K, img2)
    cam3 = Camera(3, K, img3)
    cam1.calculate_features()
    cam2.calculate_features()
    cam3.calculate_features()
    (kp_1, kp_2), (_, _) = correspondence(cam1.features, cam2.features, 0.99)

    # (R_W_C2, T_W_C2), P = estimate_next_camera_pose(cam1, kp_1, kp_2)
    # cam2.rotation = R_W_C2
    # cam2.translation = T_W_C2

    # (kp_2, kp_3), (_, _) = correspondence(
    #     cam2.features, cam3.features, 0.99
    # )

    # (R_W_C2, T_W_C2), P = estimate_next_camera_pose(cam2, kp_2, kp_3)
    # cam3.rotation = R_W_C2
    # cam3.translation = T_W_C2

    # cameras = []
    # cameras.append(cam1)
    # cameras.append(cam2)
    # cameras.append(cam3)
    # plot_plotly(P, cameras)

    t1 = time.time()

    p1 = np.hstack([kp_1, np.ones((kp_1.shape[0], 1))]).T
    p2 = np.hstack([kp_2, np.ones((kp_2.shape[0], 1))]).T

    print(f"p1: {p1.shape}")
    # mask_f contains the indices of all inlier points obtained from RANSAC
    F, mask_f = get_fundamental_matrix(p1[:2, :].T, p2[:2, :].T)
    E = essential_matrix_from_fundamental_matrix(F, K)

    # selecting only the inlier points (from RAMSAC mask)
    p1 = p1[:, mask_f.ravel() == 1]
    p2 = p2[:, mask_f.ravel() == 1]

    # find rotation and translation from camera 2 to camera 1
    R_C2_C1, T_C2_C1 = decomposeEssentialMatrix(E)
    t2 = time.time()
    # Make sure that the points are in front of the camera
    R_C2_C1, T_C2_C1 = disambiguateRelativePose(R_C2_C1, T_C2_C1, p1, p2, K, K)
    t3 = time.time()

    print(f"Time to calculate without disambiguateRelativePose: {t2-t1}")
    print(f"Time to calculate disambiguateRelativePose: {t3-t2}")
    print(f"Roataion: {R_C2_C1}")
    print(f"Translation: {T_C2_C1 }")

    # compute the camera projection matrices
    M1 = K @ np.eye(3, 4)
    # we can do this as camera 1 is at the origin and orientation of world frame
    M2 = K @ np.c_[R_C2_C1, T_C2_C1]
    print(f"Camera matrix 1: {M1}")
    print(f"Camera matrix 2: {M2}")
    P = linearTriangulation(p1, p2, M1, M2)

    print(f"Shape of P: {P}")
    # btw if you want to use opencv you have to dehomogenize the points
    # P = cv.triangulatePoints(M1, M2, p1[:2, :], p2[:2, :])
    # P[:3, :] /= P[3, :]

    # removing all points behind the camera and that are to far away
    # TODO: add params for closest and farthest point threshold
    mask = P[2, :] > 0
    P = P[:, mask]
    mask = P[2, :] < 100
    P = P[:, mask]

    cameras = []
    print(f"Camera 1: {M1}")
    print(f"Camera 2: {M2}")

    # this is R @ -T = C2_W_Center
    # get the center of camera 2 in world coordinates, again world coordinates are the same as camera 1
    T_W_C2 = -R_C2_C1.T @ T_C2_C1
    T_C2_C1 = T_C2_C1.reshape((3, 1))

    # rotate camera 1
    R_W_C1 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    cam1.R_to_world = R_W_C1.T
    cam2.R_to_world = (R_W_C1 @ R_C2_C1.T).T
    cam2.translation = T_C2_C1

    print(f"Center of camera 2 in world coordinates: {T_W_C2}")
    cameras.append(cam1)
    cameras.append(cam2)
    print(P.shape)
    R = np.eye(4)
    R[:3, :3] = cam1.R_to_world
    P = R @ P

    plot_plotly(P, cameras)
