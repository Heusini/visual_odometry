from typing import List
import numpy as np
from state import Transform3D
from draw_camera_wireframe import draw_camera_wireframe

def plot_points_cameras(Ps : List[np.ndarray], to_world_transforms: List[Transform3D]):
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

    for i, transform in enumerate(to_world_transforms):
        centerpoint = transform.tR() @ np.array([0, 0, 0, 1])
        print(centerpoint)
        scatter_2D = go.Scatter(
            x=[centerpoint[0]],
            y=[centerpoint[2]],
            mode="markers+text",
            textposition="bottom center",
            text=[f"Cam: {i}"],
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
    for i, transform in enumerate(to_world_transforms):
        camera_wireframe = draw_camera_wireframe(
            transform, 0.5, 0.5, f"Cam: {i}", colors[count]
        )
        count = (count + 1) % len(colors)
        for line in camera_wireframe:
            plotly_fig.add_trace(line, 1, 2)

    transform = dict(
        up=dict(x=0, y=-1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=-2, z=-2)
    )

    plotly_fig.update_layout(scene_camera=transform, title='default')
    plotly_fig.show()