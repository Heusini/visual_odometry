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
        'keypoints_prev' : kp_a,
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
            storage_type='local', 
            data=state),
        dcc.Interval(
            id='interval-component',
            interval=200, # in milliseconds
            n_intervals=0
        ),
    ])

    app.run(debug=True, use_reloader=True)

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
    kp_prev = np.array(data['keypoints_prev'])
    landmarks = np.array(data['landmarks'])
    img = np.array(data['img'])
    camera_translation = np.array(data['camera_translation'])
    camera_rotation = np.array(data['camera_rotation'])

    img_x = img.shape[1]
    img_y = img.shape[0]

    # Create the graph with subplots
    fig = plotly.tools.make_subplots(rows=1, cols=2, vertical_spacing=0.2)
    # plot some random data
    # invert y axis of kps and image
    kp[:, 1] = img_y - kp[:, 1]
    kp_prev[:, 1] = img_y - kp_prev[:, 1]
    img = np.flip(img, axis=0)
    fig.append_trace(go.Scatter(
        x=kp_prev[:, 0],
        y=kp_prev[:, 1],
        name='Previous Keypoints',
        mode='markers',
        # marker stule here
        marker=dict(
            size=3,
            color='rgba(0, 0, 255, .8)',
        ),
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=kp[:, 0],
        y=kp[:, 1],
        name='Keypoints',
        mode='markers',
        # marker stule here
        marker=dict(
            size=4,
            color='rgba(255, 0, 0, .8)',
        ),
    ), row=1, col=1)

    # add image behind the plot
    fig.append_trace(go.Heatmap(
        z=img,
        colorscale='gray',
        showscale=False,
    ), row=1, col=1)

    # set the limits of the plot to the limits of the data
    fig.update_xaxes(range=[0, img_x], row=1, col=1)
    fig.update_yaxes(range=[0, img_y], row=1, col=1)

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