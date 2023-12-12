import os
import numpy as np
import cv2

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback, State
import plotly

from continuous import process_frame, FrameState
from correspondence import correspondence
from sfm import sfm
import helpers

# Multiple components can update everytime interval gets fired.
@callback(
    Output('update_graph', 'figure'),
    Input('interval-component', 'n_intervals'),
    [
        State("badges", "children"), 
        State("session", "data")
    ],
)
def update_graph(n, state, session_cache):

    print(state)

    if n == 1:
        init_state = init_state()


    # Create the graph with subplots
    fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    fig.append_trace({
        'x': frame_state.keypoints[:,0],
        'y': frame_state.keypoints[:,1],
        'name': 'Altitude',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)

    # fig.append_trace({
    #     'x': data['Longitude'],
    #     'y': data['Latitude'],
    #     'text': data['time'],
    #     'name': 'Longitude vs Latitude',
    #     'mode': 'lines+markers',
    #     'type': 'scatter'
    # }, 2, 1)

    return fig

def init_state() -> FrameState:
    DATAPATH = 'data/kitti/05/image_0/'
    I_0 = 1
    I_1 = 4    

    K = np.array(
        [
            [7.188560000000e02, 0, 6.071928000000e02],
            [0, 7.188560000000e02, 1.852157000000e02],
            [0, 0, 1],
        ]
    )

    imgs = helpers.load_images(DATAPATH)

    print(imgs.shape)
    imgs_subarray = imgs[I_0:I_1]

    (kp_a, kp_b), (des_a, des_b) = correspondence(imgs_subarray)

    print(kp_a.shape, kp_b.shape)

    landmarks, camera_a, camera_b = sfm(kp_a, kp_b, K)

    return FrameState(I_1, kp_b, des_b, landmarks, camera_b)

def pipeline():
    app = Dash(__name__)
    app.layout = html.Div([
        html.H4('TERRA Satellite Live Feed'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        ),
        # The memory store reverts to the default on every page refresh
        dcc.Store(id="memory"),
        # The local store will take the initial data
        # only the first time the page is loaded
        # and keep it until it is cleared.
        dcc.Store(id="local", storage_type="local"),
        # Same as the local store but will lose the data
        # when the browser/tab closes.
        dcc.Store(id="session", storage_type="session"),
    ])

    app.run(debug=True)


    # continuous operation
    for i in range(I_1 + 1, imgs.shape[0]):
        print(f"Processing frame {i}")
        frame_state = process_frame(frame_state, imgs[i], K)


if __name__ == '__main__':
    pipeline()