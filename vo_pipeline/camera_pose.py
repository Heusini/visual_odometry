from typing import List

import cv2 as cv
import numpy as np
from exercise_helpers.decompose_essential_matrix import decomposeEssentialMatrix
from exercise_helpers.disambiguate_relative_pose import disambiguateRelativePose
from exercise_helpers.normalise_2D_pts import normalise2DPts


class Camera:
    def __init__(self, rotation, translation, K, name):
        self.name = name
        self.rotation = rotation
        self.translation = translation
        self.K = K


def get_fundamental_matrix(keypoints_a, keypoints_b):
    fundamental_mat, mask = cv.findFundamentalMat(
        keypoints_a, keypoints_b, cv.RANSAC, 3, 0.99
    )
    return fundamental_mat, mask


def get_essential_matrix(point_a, point_b, K):
    essential_mat, _ = cv.findEssentialMat(point_a, point_b, K, cv.RANSAC, 0.99, 1)
    return essential_mat


def essential_matrix_from_fundamental_matrix(fundamental_mat, K):
    return K.T @ fundamental_mat @ K


def plot_plotly(P, cameras: List[Camera]):
    import plotly.graph_objects as go

    plotly_fig = go.Figure()
    scatter_3d = go.Scatter3d(
        x=P[0, :],
        y=P[1, :],
        z=P[2, :],
        mode="markers",
        marker=dict(
            size=1,
            color="red",  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            opacity=0.8,
        ),
    )
    plotly_fig.add_trace(scatter_3d)
    colors = ["black", "green", "blue", "yellow", "orange", "purple", "pink", "brown"]
    count = 0
    for camera in cameras:
        camera_wireframe = draw_camera_wireframe(
            camera.rotation,
            camera.translation,
            0.5,
            0.5,
            camera.name,
            colors[count],
        )
        count = (count + 1) % len(colors)
        for line in camera_wireframe:
            plotly_fig.add_trace(line)
    plotly_fig.show()


def draw_camera_wireframe(rotation, translation, f, size, cam_name, color="black"):
    import plotly.graph_objects as go

    p1_c = np.array([-size / 2, -size / 2, f])
    p2_c = np.array([size / 2, -size / 2, f])
    p3_c = np.array([size / 2, size / 2, f])
    p4_c = np.array([-size / 2, size / 2, f])

    p1_w = rotation.T @ (p1_c - translation[:, 0])
    p2_w = rotation.T @ (p2_c - translation[:, 0])
    p3_w = rotation.T @ (p3_c - translation[:, 0])
    p4_w = rotation.T @ (p4_c - translation[:, 0])

    center_point = rotation.T @ (-translation[:, 0])
    print(center_point.shape)

    # draw camera wireframe
    camera_wireframe = go.Scatter3d(
        x=[p1_w[0], p2_w[0], p3_w[0], p4_w[0], p1_w[0]],
        y=[p1_w[1], p2_w[1], p3_w[1], p4_w[1], p1_w[1]],
        z=[p1_w[2], p2_w[2], p3_w[2], p4_w[2], p1_w[2]],
        mode="lines",
        name=cam_name,
        line=dict(color=color, width=4),
        legendgroup=cam_name,
        showlegend=True,
    )

    center_line1 = go.Scatter3d(
        x=[center_point[0], p1_w[0]],
        y=[center_point[1], p1_w[1]],
        z=[center_point[2], p1_w[2]],
        mode="lines",
        name=cam_name + "line1",
        line=dict(color=color, width=4),
        legendgroup=cam_name,
        showlegend=False,
    )
    center_line2 = go.Scatter3d(
        x=[center_point[0], p2_w[0]],
        y=[center_point[1], p2_w[1]],
        z=[center_point[2], p2_w[2]],
        mode="lines",
        name=cam_name + "line2",
        line=dict(color=color, width=4),
        legendgroup=cam_name,
        showlegend=False,
    )
    center_line3 = go.Scatter3d(
        x=[center_point[0], p3_w[0]],
        y=[center_point[1], p3_w[1]],
        z=[center_point[2], p3_w[2]],
        mode="lines",
        name=cam_name + "line3",
        line=dict(color=color, width=4),
        legendgroup=cam_name,
        showlegend=False,
    )
    center_line4 = go.Scatter3d(
        x=[center_point[0], p4_w[0]],
        y=[center_point[1], p4_w[1]],
        z=[center_point[2], p4_w[2]],
        mode="lines",
        name=cam_name + "line4",
        line=dict(color=color, width=4),
        legendgroup=cam_name,
        showlegend=False,
    )

    lines = [camera_wireframe, center_line1, center_line2, center_line3, center_line4]

    return lines


if __name__ == "__main__":
    import os
    import sys
    import time

    from correspondence import correspondence
    from exercise_helpers.camera_3d import drawCamera
    from exercise_helpers.linear_triangulation import linearTriangulation
    from helpers import load_images

    imgs = []
    # imgs = load_images("data/kitti/05/image_0/", start=0, end=2)
    # img1 = cv.imread("data/kitti/05/image_0/000000.png")
    # img2 = cv.imread("data/kitti/05/image_0/000004.png")
    img1 = cv.imread("data/parking/images/img_00000.png")
    img2 = cv.imread("data/parking/images/img_00003.png")
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    imgs.append(img1)
    imgs.append(img2)
    K_kitty = np.array(
        [
            [7.188560000000e02, 0, 6.071928000000e02],
            [0, 7.188560000000e02, 1.852157000000e02],
            [0, 0, 1],
        ]
    )
    K_parking = np.array([[331.37, 0, 320], [0, 369.568, 240], [0, 0, 1]])
    K = K_parking
    imgs = np.asarray(imgs)
    (keypoints_a, keypoints_b), (_, _) = correspondence(imgs, 0.99)

    t1 = time.time()

    p1 = np.hstack([keypoints_a, np.ones((keypoints_a.shape[0], 1))]).T
    p2 = np.hstack([keypoints_b, np.ones((keypoints_b.shape[0], 1))]).T

    print(f"p1: {p1.shape}")
    F, _ = get_fundamental_matrix(p1[:2, :].T, p2[:2, :].T)
    E = essential_matrix_from_fundamental_matrix(F, K)

    Rots, u3 = decomposeEssentialMatrix(E)
    t2 = time.time()
    R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, p1, p2, K, K)
    # retval, R, t, mask = cv.recoverPose(E, p1, p2, K)
    # R_C2_W = R
    # T_C2_W = t
    t3 = time.time()

    print(f"Time to calculate without disambiguateRelativePose: {t2-t1}")
    print(f"Time to calculate disambiguateRelativePose: {t3-t2}")
    print(f"Roataion: {R_C2_W}")
    print(f"Translation: {T_C2_W}")

    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R_C2_W, T_C2_W]
    print(f"Camera matrix 1: {M1}")
    print(f"Camera matrix 2: {M2}")
    P = linearTriangulation(p1, p2, M1, M2)

    cameras = []
    print(f"Camera 1: {M1}")
    print(f"Camera 2: {M2}")

    # this is R @ -T = C2_W_Center
    T_W_C2 = -R_C2_W.T @ T_C2_W
    T_C2_W = T_C2_W.reshape((3, 1))
    print(f"Center of camera 2 in world coordinates: {T_W_C2}")
    cameras.append(Camera(np.eye(3, 3), np.zeros((3, 1)), K, "Cam 1"))
    cameras.append(Camera(R_C2_W, T_C2_W, K, "Cam 2"))

    plot_plotly(P, cameras)

    # # plot keypoints from top view and camera
    # import matplotlib.pyplot as plt
    # from exercise_helpers.arrow_3d import Arrow3D

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(P[0, :], P[1, :], P[2, :], marker="o", s=4)
    # ax.set_xlabel("X Label")
    # ax.set_ylabel("Y Label")
    # ax.set_zlabel("Z Label")
    #
    # # Display camera pose
    # drawCamera(ax, np.zeros((3,)), np.eye(3), length_scale=2)
    # ax.text(-0.1, -0.1, -0.1, "Cam 1")
    #
    # center_cam2_W = -R_C2_W.T @ T_C2_W
    # drawCamera(ax, center_cam2_W, R_C2_W.T, length_scale=2)
    # ax.text(
    #     center_cam2_W[0] - 0.1, center_cam2_W[1] - 0.1, center_cam2_W[2] - 0.1, "Cam 2"
    # )
    #
    # plt.show()
