from twoDtwoD import twoDtwoD
from utils.dataloader import DataLoader, Dataset
from initialization import initialize
from pnp import pnp
from tracking import Tracking, TrackManager
import numpy as np
import cv2 as cv

# Constants
init_bandwidth = 4

def main():
    # select dataset
    dataset = Dataset.KITTI
    
    # load data
    # if you remove steps then all the images are used
    loader = DataLoader(dataset, start=105, stride=1, steps=90)
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
    track_manager = TrackManager(
        angle_threshold=2.5*np.pi/180,
        same_keypoints_threshold=0.5,
        max_track_length=10,
    )
    t = 0
    while t < len(states) - 3:
        if t == 50:
            break
        
        # computes keypoints, landmarks and pose for step+1 frame, given step frame
        # Uses KLT to track keypoints, meaning the set of keypoints in step+1 is included in
        # the set of keypoints in step frame, is however <= in size
        states[t+1] = pnp(states[t], states[t+1], K)
        if DEBUG:
            # add new keypoints for current frame t
            # tracking.add_keypoint_candidates(states[step].features.keypoints, start_frame_index=states[step].t, current_state=states[step])
            track_manager.start_new_track(states[t])

            track_manager.update(
                t,
                img_i=cv.imread(states[t].img_path), 
                img_j=cv.imread(states[t+1].img_path))
            
            landmarks, keypoints = track_manager.get_new_landmarks(
                t + 1,
                min_track_length=5,
                frame_states=states,
                K = K)
            
            states[t+1].landmarks = np.hstack([states[t+1].landmarks, landmarks])
            states[t+1].keypoints = np.hstack([states[t+1].keypoints, keypoints])

            # tracking.plot_stats(states[step])
        else:
            if states[t].landmarks.shape[1] / states[0].landmarks.shape[1] < 0.2:
                initialize(states[t], states[t + 3], K)

        t += 1

    plot_points_cameras(
        [],
        [state.cam_to_world for state in states[0:t-1]],
        plot_points=False
    )

if __name__ == '__main__':
    from plot_points_cameras import plot_points_cameras
    main()