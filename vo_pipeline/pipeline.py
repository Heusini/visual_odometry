import os
import numpy as np
import cv2

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import plotly

from continuous import process_frame, FrameState
from correspondence import correspondence
from sfm import sfm
import helpers

# Multiple components can update everytime interval gets fired.
@callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(frame_state : FrameState):

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

def pipeline():
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

    landmarks, camera_a, camera_b = sfm(kp_a, kp_b, K)

    app = Dash(__name__)
    app.layout = html.Div([
        html.H4('TERRA Satellite Live Feed'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )
    ])

    frame_state = FrameState(I_1, K, kp_b, des_b, landmarks, camera_b)

    # continuous operation
    for i in range(I_1 + 1, imgs.shape[0]):
        frame_state = process_frame(frame_state, imgs[i])


if __name__ == '__main__':
    pipeline()