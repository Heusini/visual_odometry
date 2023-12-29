import cv2 as cv
import numpy as np
from state import FrameState
from utils.linear_triangulation import linearTriangulation
from klt import klt
from typing import List, Mapping
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Track:
    start_t : int
    end_t : int
    keypoints_start : np.ndarray # [N , 2] where N is the number of keypoints
    keypoints_end : np.ndarray # [N , 2] where N is the number of keypoints

    def __init__(self, start_t: int, start_keypoints: np.ndarray) -> None:
        self.start_t = start_t
        self.keypoints_start = start_keypoints
        self.keypoints_end = start_keypoints
        self.end_t = start_t

    def track(self, img_i: np.ndarray, img_j: np.ndarray):
        keypoints_next, mask = klt(self.keypoints_end[ : , :], img_i, img_j)
        print(f"KLT: tracking {np.sum(mask)} keypoints from track starting at {self.start_t}")
        keypoints_next = np.squeeze(keypoints_next)
        keypoints_next = keypoints_next[mask, :]
        self.keypoints_start = self.keypoints_start[mask, :]
        self.keypoints_end = keypoints_next
        self.end_t += 1

    def size(self):
        return self.keypoints_end.shape[0]

    def length(self):
        return self.end_t - self.start_t

    def get_kp_at_time(self, t: int):
        index = t - self.start_t
        if index < 0 or index >= self.length():
            raise Exception("Invalid time index")
        
        return self.keypoints[index, :, :]

class TrackManager:
    active_tracks : Mapping[int, Track] # maps from start frame index to track
    inactive_tracks : Mapping[int, Track] # maps from start frame index to track
    angle_threshold : float
    same_keypoints_threshold : float
    max_track_length : int

    def __init__(
        self, 
        angle_threshold: float,
        same_keypoints_threshold: float,
        max_track_length: int
    ) -> None:
        
        self.active_tracks = {}
        self.angle_threshold = angle_threshold
        self.same_keypoints_threshold = same_keypoints_threshold
        self.max_track_length = max_track_length

    def start_new_track(
        self, 
        state: FrameState,
        check_keypoints: bool = True
    ):
        new_kps = state.features.get_positions()
        if check_keypoints:
            landmarks_kps = state.keypoints
            mask_x = self.threshold_mask(new_kps[:, 0], landmarks_kps[:, 0])
            mask_y = self.threshold_mask(new_kps[:, 1], landmarks_kps[:, 1])

            mask = np.logical_not(np.logical_and(mask_x, mask_y))
            
            self.active_tracks[state.t] = Track(state.t, new_kps[mask, :])
        else:
            self.active_tracks[state.t] = Track(state.t, new_kps)

    def threshold_mask(self, arr1, arr2):
        arr1 = np.atleast_1d(arr1)
        arr2 = np.atleast_1d(arr2)

        # Calculate the absolute differences and check if within tolerance
        diffs = np.abs(arr1[:, None] - arr2)
        return np.any(diffs <= self.same_keypoints_threshold, axis=1)

    def update(
        self, 
        t : int, 
        img_i: np.ndarray, 
        img_j: np.ndarray
    ):
        for track_start_t in list(self.active_tracks.keys()):
            # remove tracks that are too long or have no keypoints left
            if (self.active_tracks[track_start_t].length() > self.max_track_length or \
                self.active_tracks[track_start_t].size() == 0):
                
                del self.active_tracks[track_start_t]
                continue

            self.active_tracks[track_start_t].track(img_i, img_j)

    def get_new_landmarks(
        self, 
        time_j : int, 
        min_track_length: int, 
        frame_states: List[FrameState], 
        K: np.ndarray,
        compare_to_landmarks: bool = True
    ) -> (np.ndarray, np.ndarray):
        landmarks = np.zeros((3, 0))
        keypoints = np.zeros((2, 0))

        for time_i in list(self.active_tracks.keys()):
            # skip tracks that are too short
            if self.active_tracks[time_i].length() < min_track_length:
                continue

            track = self.active_tracks[time_i]

            state_i = frame_states[time_i]
            state_j = frame_states[time_j]

            M_cami_to_camj = np.linalg.inv(state_i.cam_to_world) @ state_j.cam_to_world

            kp_i = np.vstack([track.keypoints_start.T, np.ones((1, track.size()))])
            kp_j = np.vstack([track.keypoints_end.T, np.ones((1, track.size()))])
            P_cami = linearTriangulation(
                kp_i, 
                kp_j, 
                K @ np.eye(3, 4),
                K @ M_cami_to_camj[0:3, :]
            )[:3, :]

            #filter points behind camera and far away
            max_distance = 100
            mask_distance = np.logical_and(
                P_cami[2, :] > 0, np.abs(np.linalg.norm(P_cami, axis=0)) < max_distance)
            P_cami = P_cami[:, mask_distance]
            kp_j = kp_j[:, mask_distance]

            P_world = state_i.cam_to_world @ np.vstack([P_cami, np.ones((1, P_cami.shape[1]))])
            
            # remove landmarks where the angle between the two cameras is too small
            T_cami = state_i.cam_to_world[:3, 3].reshape(-1, 1)
            T_camj = state_j.cam_to_world[:3, 3].reshape(-1, 1)

            P_to_cami = P_world[:3, :] - T_cami
            P_to_camj = P_world[:3, :] - T_camj
            dot = np.diag(np.dot(P_to_cami.T, P_to_camj))

            angles = np.arccos(np.clip(dot, -1.0, 1.0))

            angle_mask = angles > self.angle_threshold

            print(f"filtering out {np.sum(np.logical_not(angle_mask))} landmarks that have too small angle between cameras")

            P_world = P_world[:, angle_mask]
            kp_j = kp_j[:, angle_mask]

            # remove landmarks that are too close to existing landmarks
            if compare_to_landmarks:
                mask_x = self.threshold_mask(P_world[0, :], state_j.landmarks[0, :])
                mask_y = self.threshold_mask(P_world[1, :], state_j.landmarks[1, :])
                mask_z = self.threshold_mask(P_world[2, :], state_j.landmarks[2, :])

                mask_pos = np.logical_not(np.logical_and(np.logical_and(mask_x, mask_y), mask_z))

                print(f"filtering out {np.sum(np.logical_not(mask_pos))} landmarks that are too close to existing landmarks")

                P_world = P_world[:, mask_pos]
                kp_j = kp_j[:, mask_pos]

            # state_j.landmarks = np.hstack([state_j.landmarks, P_world[:3, :]])
            # state_j.keypoints = np.hstack([state_j.keypoints, kp_j[:2, :]])

            del self.active_tracks[time_i]

            landmarks = np.hstack([landmarks, P_world[:3, :]])
            keypoints = np.hstack([keypoints, kp_j[:2, :]])

        return landmarks, keypoints

    def plot_stats(self, current_state: FrameState):
        if len(list(self.tracks.keys())) > 0:
            # first plot
            self.num_keypoints = [self.tracks[key][0].shape[0] for key in list(self.tracks.keys())]
            keys = list(self.tracks.keys())
            colors = cm.viridis(np.linspace(0, 1, len(keys)))
            self.ax.clear()
            self.ax.bar(keys, self.num_keypoints, color=colors)
            self.ax.set_xlabel('Keys')
            self.ax.set_ylabel('List Lengths')
            self.ax.set_title('List Lengths in Dictionary Over Time')
            
            # second plot
            self.ax1.plot(current_state.t, current_state.landmarks.shape[1], 'o-')
            self.ax1.set_xlabel('frame k')
            self.ax1.set_ylabel('Number of landmarks at k')
            self.ax1.set_title('Tracking the number of landmarks')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.5)


if __name__ == "__main__":
    from utils.dataloader import DataLoader, Dataset
    from initialization import initialize
    from plot_points_cameras import plot_points_cameras
    from twoDtwoD import FeatureDetector
    from pnp import pnp

    # select dataset
    dataset = Dataset.KITTI
    
    # load data
    # if you remove steps then all the images are used
    loader = DataLoader(dataset, start=0, stride=1, steps=100)
    print("Loading data...")

    loader.load_data()
    print("Data loaded!")

    K, poses, states = loader.get_data()
    print("Data retrieved!")

    # computes landmarks and pose for the first two frames
    if dataset == Dataset.KITTI:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]
    elif dataset == Dataset.PARKING:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]
    elif dataset == Dataset.MALAGA:
        initialize(states[0], states[4], K)
        init_states = [states[0], states[4]]
    elif dataset == Dataset.WOKO:
        initialize(states[0], states[3], K)
        init_states = [states[0], states[3]]

    track_manager = TrackManager(
        angle_threshold=2.5*np.pi/180,
        same_keypoints_threshold=0.5,
        max_track_length=10,
    )

    for t in range(40):

        states[t+1] = pnp(states[t], states[t+1], K)

        track_manager.start_new_track(
            states[t],
            check_keypoints=False)

        track_manager.update(
            t,
            img_i=cv.imread(states[t].img_path), 
            img_j=cv.imread(states[t+1].img_path))
        
        landmarks, keypoints = track_manager.get_new_landmarks(
            t+1,
            min_track_length=5,
            frame_states=states,
            K=K,
            compare_to_landmarks=False)
       
        if t > 10:
            # plot image using cv
            img = cv.imread(states[t].img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # plot keypoint tracks using cv2
            n_tracks = len(track_manager.active_tracks)
            for track_index, track in enumerate(track_manager.active_tracks.values()):
                start = track.keypoints_start.astype(int)
                end = track.keypoints_end.astype(int)

                for i in range(track.size()):
                    # above line plot code but add to ax[0]
                    cv.line(
                        img,
                        (start[i, 0], start[i, 1]),
                        (end[i, 0], end[i, 1]),
                        color=(
                            255 * track_index / n_tracks,
                            255 * ( 1 - track_index / n_tracks),
                            0),
                        thickness=1
                    )

            # plot keypoints using cv2
            print(keypoints.shape)
            for i in range(keypoints.shape[1]):
                cv.circle(
                    img,
                    (int(keypoints[0, i]), int(keypoints[1, i])),
                    radius=3,
                    color=(0, 255, 0),

                    thickness=3
                )

            cv.imshow('image',img)
            cv.waitKey()
