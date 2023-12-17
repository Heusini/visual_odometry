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


def plot_plotly(P, cameras: List[Camera]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plotly_fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scatter"}, {"type": "scatter3d"}]]
    )
    # draw 2D xz scatter plot plus camera position
    scatter_2D = go.Scatter(
        x=P[0, :],
        y=P[1, :],
        mode="markers",
        marker=dict(
            size=3,
            color="red",  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            opacity=0.8,
        ),
    )
    plotly_fig.add_trace(scatter_2D, 1, 1)

    center_point_C1 = cameras[0].rotation.T @ (-cameras[0].translation[:, 0])
    center_point_C2 = cameras[1].rotation.T @ (-cameras[1].translation[:, 0])
    camera_poses = go.Scatter(
        x=[center_point_C1[0], center_point_C2[0]],
        y=[center_point_C1[2], center_point_C2[2]],
        mode="markers+text",
        textposition="bottom center",
        text=["C1", "C2"],
        marker=dict(size=5, color="blue", colorscale="Viridis", opacity=0.8, symbol=4),
    )
    plotly_fig.add_trace(camera_poses, 1, 1)
    # draw 3D scatter plot plus camera frames
    scatter_3d = go.Scatter3d(
        x=P[0, :],
        y=P[1, :],
        z=P[2, :],
        mode="markers",
        marker=dict(
            size=3,
            color="red",  # set color to an array/list of desired values
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
        camera_wireframe = draw_camera_wireframe(
            camera.rotation,
            camera.translation,
            0.5,
            0.5,
            f"Cam: {camera.id}",
            colors[count],
        )
        count = (count + 1) % len(colors)
        for line in camera_wireframe:
            plotly_fig.add_trace(line, 1, 2)
    plotly_fig.show()


# test stuff
if __name__ == "__main__":
    imgs = []
    # imgs = load_images("data/kitti/05/image_0/", start=0, end=2)
    dataset = DataSetEnum.PARKING

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
        K = K_kitty
    elif dataset == DataSetEnum.PARKING:
        img1 = cv.imread("data/parking/images/img_00000.png")
        img2 = cv.imread("data/parking/images/img_00003.png")
        K = K_parking
    elif dataset == DataSetEnum.MALAGA:
        pass

    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    imgs = np.asarray(imgs)
    cam1 = Camera(1, K, img1)
    cam2 = Camera(2, K, img2)
    cam1.calculate_features()
    cam2.calculate_features()
    (keypoints_a, keypoints_b), (_, _) = correspondence(
        cam1.features, cam2.features, 0.99
    )

    t1 = time.time()

    p1 = np.hstack([keypoints_a, np.ones((keypoints_a.shape[0], 1))]).T
    p2 = np.hstack([keypoints_b, np.ones((keypoints_b.shape[0], 1))]).T

    print(f"p1: {p1.shape}")
    # mask_f contains the indices of all inlier points obtained from RANSAC
    F, mask_f = get_fundamental_matrix(p1[:2, :].T, p2[:2, :].T)
    E = essential_matrix_from_fundamental_matrix(F, K)

    # selecting only the inlier points
    p1 = p1[:, mask_f.ravel() == 1]
    p2 = p2[:, mask_f.ravel() == 1]

    Rots, u3 = decomposeEssentialMatrix(E)
    t2 = time.time()
    R_C2_C1, T_C2_C1 = disambiguateRelativePose(Rots, u3, p1, p2, K, K)
    t3 = time.time()

    print(f"Time to calculate without disambiguateRelativePose: {t2-t1}")
    print(f"Time to calculate disambiguateRelativePose: {t3-t2}")
    print(f"Roataion: {R_C2_C1}")
    print(f"Translation: {T_C2_C1}")

    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R_C2_C1, T_C2_C1]
    print(f"Camera matrix 1: {M1}")
    print(f"Camera matrix 2: {M2}")
    P = linearTriangulation(p1, p2, M1, M2)
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
    T_W_C2 = -R_C2_C1.T @ T_C2_C1
    T_C2_C1 = T_C2_C1.reshape((3, 1))

    R_W_C1 = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])

    cam1.rotation = R_W_C1 @ cam1.rotation
    cam2.rotation = R_W_C1 @ R_C2_C1.T
    cam2.translation = T_C2_C1

    print(f"Center of camera 2 in world coordinates: {T_W_C2}")
    cameras.append(cam1)

    cameras.append(cam2)
    print(P.shape)
    R = np.eye(4)
    R[:3, :3] = cam1.rotation
    P = R @ P

    plot_plotly(P, cameras)
