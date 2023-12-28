from twoDtwoD import twoDtwoD
from utils.dataloader import DataLoader, Dataset
from initialization import initialize
from pnp import pnp
from tracking import Tracking
import numpy as np
import cv2 as cv

# Constants
init_bandwidth = 4

def main():
    # select dataset
    dataset = Dataset.KITTI
    
    # load data
    # if you remove steps then all the images are used
    loader = DataLoader(dataset, start=90, stride=1, steps=90)
    print("Loading data...")

    loader.load_data()
    print("Data loaded!")

    K, poses, states = loader.get_data()
    print("Data retrieved!")
    
    # TODO: use some sort of params variable to store the best parameters
    # for each dataset; Maybe add this to the dataloader
    
    # computes landmarks and pose for the first two frames
    if dataset == Dataset.KITTI:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]
        init_frame_indices = [0, 3]
    elif dataset == Dataset.PARKING:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]
        init_frame_indices = [0, 3]
    elif dataset == Dataset.MALAGA:
        initialize(states[0], states[4], K)
        init_states = [states[0], states[4]]
        init_frame_indices = [0, 4]
    elif dataset == Dataset.WOKO:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]
        init_frame_indices = [0, 3]

    plot_points_cameras(
        [init_states[0].landmarks, init_states[1].landmarks],
        [init_states[0].cam_to_world, init_states[1].cam_to_world]
    )
    
    # TODO: Implement Continuous Part: Feel free to change
    DEBUG = True
    tracking = Tracking(angle_threshold=2.5*np.pi/180, init_frame_indices=init_frame_indices)
    step = 0
    while step < len(states) - 3:
        if step == 21:
            break
        
        if DEBUG:
            # add new keypoints for current frame t
            tracking.add_keypoint_candidates(
                states[step].features.keypoints, start_frame_index=states[step].t)

            # track existing keypoints
            tracking.track_keypoints(
                img_i=cv.imread(states[step].img_path), img_j=cv.imread(states[step+1].img_path))

            # check if new landmarks can be created and if so do it
            tracking.check_for_new_landmarks(frame_states=states, K=K)
        else:
            if states[step].landmarks.shape[1] / states[0].landmarks.shape[1] < 0.2:
                initialize(states[step], states[step + 3], K)

        # calculate the camera pose from the information given at frame i and i + 1
        pnp(states[step], states[step+1], K)
        step += 1

    plot_points_cameras(
        [state.landmarks for state in states[0:step-1]],
        [state.cam_to_world for state in states[0:step-1]],
        plot_points=True
    )

if __name__ == '__main__':
    from plot_points_cameras import plot_points_cameras
    main()