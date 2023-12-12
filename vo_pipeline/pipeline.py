import os
import numpy as np
import cv2

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback, State
import plotly

from continuous import process_frame
from correspondence import correspondence
from sfm import sfm
import helpers

DATAPATH = 'data/kitti/05/image_0/'

def init_state() -> dict:
    I_0 = 1
    I_1 = 4    

    K = np.array(
        [
            [7.188560000000e02, 0, 6.071928000000e02],
            [0, 7.188560000000e02, 1.852157000000e02],
            [0, 0, 1],
        ]
    )

    imgs = helpers.load_images(DATAPATH, start=I_0, end=I_1)

    (kp_a, kp_b), (des_a, des_b) = correspondence(imgs, 0.9)

    landmarks, camera_a, camera_b = sfm(kp_a, kp_b, K)

    return {
        'K' : K,
        'img' : imgs[-1],
        'iteration' : 0,
        'keypoints' : kp_b,
        'descriptors' : des_b,
        'landmarks' : landmarks,
        'camera_rotation' : camera_b.rotation,
        'camera_translation' : camera_b.translation,
    }

def pipeline():
    print("Initializing state...")
    state = init_state()
    print("State initialized.")
    app = Dash(__name__)
    app.layout = html.Div([
        html.H4('VISUAL ODOMETRY PIPELINE'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='update_graph'),
        dcc.Store(
            id="frame_state", 
            storage_type='memory', 
            data=state),
        dcc.Interval(
            id='interval-component',
            interval=1000, # in milliseconds
            n_intervals=0
        ),
    ])

    app.run(debug=True)


    # # continuous operation
    # for i in range(I_1 + 1, imgs.shape[0]):
    #     print(f"Processing frame {i}")
    #     frame_state = process_frame(frame_state, imgs[i], K)


# Multiple components can update everytime interval gets fired.
@callback(
    [ # output
        Output('update_graph', 'figure'),
        Output('frame_state', 'data'),
    ],
    [ # input
        Input('interval-component', 'n_intervals'),
        # Input('frame_state', 'data'),
    ],
    [
        State("frame_state", "data")
    ],
)
def update_graph(n, data):

    kp = np.array(data['keypoints'])
    landmarks = np.array(data['landmarks'])
    img = np.array(data['img'])
    camera_translation = np.array(data['camera_translation'])
    camera_rotation = np.array(data['camera_rotation'])

    # Create the graph with subplots
    fig = plotly.tools.make_subplots(rows=1, cols=2, vertical_spacing=0.2)
    # plot some random data
    fig.append_trace(go.Scatter(
        x=kp[:, 0],
        y=kp[:, 1],
        name='Keypoints',
        mode='markers',
    ), row=1, col=1)
    #plot image behind the scatter plot
    # fig.add_layout_image(
    #     dict(
    #         source=img,
    #         xref="x",
    #         yref="y",
    #         x=0,
    #         y=0,
    #         sizex=1,
    #         sizey=1,
    #         sizing="contain",
    #         opacity=0.5,
    #         layer="below",
    #     )
    # )

    # fig.append_trace(go.Scatter(
    #     x=[0, 1],
    #     y=[0, 1],
    #     name='Landmarks',
    #     mode='markers',
    # ), row=1, col=2)

    fig.append_trace(go.Scatter(
        x=landmarks[:, 0],
        y=landmarks[:, 1],
        name='Landmarks',
        mode='markers',
    ), row=1, col=2)

    # # add camera position
    # fig.append_trace(go.Scatter(
    #     x=[camera_translation[0]],
    #     y=[camera_translation[1]],
    #     name='Camera',
    #     mode='markers',
    # ), row=1, col=2)

    img_id = data['iteration']

    next_image = helpers.load_images(DATAPATH, start=img_id, end=img_id + 1)[0]
    print(next_image.shape)
    data = process_frame(data, next_image)

    return fig, data

if __name__ == '__main__':
    pipeline()