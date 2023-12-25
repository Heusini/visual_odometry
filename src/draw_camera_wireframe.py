import plotly.graph_objects as go
import numpy as np

def draw_camera_wireframe(to_world : np.ndarray, f, size, cam_name, color="black"):

    p1_c = np.array([-size / 2, -size / 2, f, 1.0])
    p2_c = np.array([size / 2, -size / 2, f, 1.0])
    p3_c = np.array([size / 2, size / 2, f, 1.0])
    p4_c = np.array([-size / 2, size / 2, f, 1.0])

    M_to_world = to_world
    p1_w = M_to_world @ p1_c
    p2_w = M_to_world @ p2_c
    p3_w = M_to_world @ p3_c
    p4_w = M_to_world @ p4_c

    center_point = M_to_world @ np.array([0, 0, 0, 1])
    print(center_point)

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

    lines = [
        camera_wireframe,
        center_line1,
        center_line2,
        center_line3,
        center_line4,
    ]

    return lines