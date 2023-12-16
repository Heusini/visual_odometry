import os
from typing import Callable, Optional

import cv2
import numpy as np


def load_images(
    path: str,
    filter: Optional[Callable[[str], bool]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> np.ndarray:
    filenames = []

    # open directory and read all filenames
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file() and (filter is None or filter(entry.name)):
                filenames.append(entry.name)

    # sort filenames
    filenames.sort()

    if start is None:
        start = 0
    if end is None:
        end = len(filenames)

    # load images
    imgs = []
    for filename in filenames[start:end]:
        img = cv2.imread(path + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)

    return np.array(imgs)


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
