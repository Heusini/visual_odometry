import cv2 as cv
import numpy as np
from state import FrameState
from utils.linear_triangulation import linearTriangulation
from klt import klt
from typing import List, Mapping
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# TODO: I propose this archtecture instead. I we can simplify the code a lot with this.
# instead of using the tracking class, we would then simply maintain a list of tracks in the main loop
class Track:
    start_t : int
    keypoints : np.ndarray # [L, N , 2] where L is the number of frames and N is the number of keypoints

    def __init__(self, start_t: int, start_keypoints: np.ndarray) -> None:
        self.start_t = start_t
        self.keypoints = np.array([start_keypoints])

    def track(self, img_i: np.ndarray, img_j: np.ndarray):
        kp_j, mask = klt(self.keypoints[-1, : , :], img_i, img_j)
        kp_j = np.squeeze(kp_j)
        kp_j = kp_j[mask, :]
        self.keypoints = self.keypoints[:, mask, :]
        self.keypoints = np.concatenate((self.keypoints, np.array([kp_j])), axis=0)

    def size(self):
        return self.keypoints.shape[1]

    def length(self):
        return self.keypoints.shape[0]

    def get_kp_at_time(self, t: int):
        index = t - self.start_t
        if index < 0 or index >= self.keypoints.shape[0]:
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
        state: FrameState
    ):
        new_kps = state.features.get_positions()
        landmarks_kps = state.keypoints
        mask_x = self.threshold_mask(new_kps[:, 0], landmarks_kps[:, 0])
        mask_y = self.threshold_mask(new_kps[:, 1], landmarks_kps[:, 1])

        mask = np.logical_not(np.logical_and(mask_x, mask_y))
        
        self.active_tracks[state.t] = Track(state.t, new_kps[mask, :])

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
            if (t - track_start_t > self.max_track_length or \
                self.active_tracks[track_start_t].size() == 0):
                
                del self.active_tracks[track_start_t]
                continue

            self.active_tracks[track_start_t].track(img_i, img_j)

    def get_new_landmarks(
        self, 
        time_j : int, 
        min_track_length: int, 
        frame_states: List[FrameState], 
        K: np.ndarray
    ):
        for time_i in list(self.active_tracks.keys()):
            # skip tracks that are too short
            if self.active_tracks[time_i].length() < min_track_length:
                continue

            track = self.active_tracks[time_i]

            state_i = frame_states[time_i]
            state_j = frame_states[time_j]

            M_cami_to_camj = np.linalg.inv(state_i.cam_to_world) @ state_j.cam_to_world

            kp_i = np.vstack([track.get_kp_at_time(time_i).T, np.ones((1, track.size()))])
            kp_j = np.vstack([track.get_kp_at_time(time_j).T, np.ones((1, track.size()))])
            
            P_cami = linearTriangulation(
                kp_i, 
                kp_j, 
                K @ np.eye(3, 4),
                K @ M_cami_to_camj[0:3, :]
            )[:3, :]

            #filter points behind camera and far away
            max_distance = 100
            mask = np.logical_and(
                P_cami[2, :] > 0, np.abs(np.linalg.norm(P_cami, axis=0)) < max_distance)
            P_cami = P_cami[:, mask]
            kp_j = kp_j[:, mask]

            P_world = state_i.cam_to_world @ np.vstack([P_cami, np.ones((1, P_cami.shape[1]))])
            # f_c = P_world[:3, :] - state_i[:3, 3].reshape(-1, 1)
            # c = P_world[:3, :] - state_j.cam_to_world[:3, 3].reshape(-1, 1)

            # dot_products = np.diag(np.dot(f_c.T, c))
            # start_norms = np.linalg.norm(f_c, axis=0)
            # end_norms = np.linalg.norm(c, axis=0)
            # norm_product = start_norms * end_norms
            # cos_angles = dot_products / norm_product
            # angles = np.array([np.arccos(cs_angle) for cs_angle in cos_angles])
            # mask_angle = np.where(angles.reshape(-1, 1) > self.angle_threshold)[0]
            
            # # TODO: Check if there are landmarks that are similar to some of P_world with np.isclose
            # ext_landmarks = np.hstack([frame_states[next_frame].landmarks, P_world[:3, :]])
            # frame_states[next_frame].landmarks = ext_landmarks

            # # updating the keypoints in each state since there have been multiple
            # # filtering steps that reduced the number of initial keypoints
            # ext_keypoints_end = np.hstack([frame_states[next_frame].keypoints, kp_j[:2, :]])
            # frame_states[next_frame].keypoints = ext_keypoints_end

            # remove used keypoints in track since it is now used as landmarks

            state_j.landmarks = np.hstack([state_j.landmarks, P_world[:3, :]])
            state_j.keypoints = np.hstack([state_j.keypoints, kp_j[:2, :]])

            print(f"prev landmarks: {state_j.landmarks.shape[1] - P_world.shape[1]}")
            print(f"Found {P_world.shape[1]} new landmarks")

            del self.active_tracks[time_i]

class Tracking:
    def __init__(
        self, 
        angle_threshold: float, 
        init_frame_indices: [int, int]
    ) -> None:

        # maps from start frame index to list of keypoints
        self.tracks = {}
        # list of start frame indices
        self.start_frame_indices = []
        self.angle_threshold = angle_threshold
        self.init_frame_indices = init_frame_indices

        # for plotting
        self.fig = plt.figure()

        # Plotting keypoints decay
        self.ax = self.fig.add_subplot(211)
        self.ax.set_xlabel('Keys')
        self.ax.set_ylabel('List Lengths')
        self.ax.set_title('List Lengths in Dictionary Over Time')

        # Plotting number of landmarks over time
        self.ax1 = self.fig.add_subplot(212)
        self.ax1.set_xlabel('frame k')
        self.ax1.set_ylabel('Number of landmarks at k')
        self.ax1.set_title('Tracking the number of landmarks')

        self.annot = self.ax1.annotate("", xy=(0,0), xytext=(20,20),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        pass
    
    def add_keypoint_candidates(
        self, 
        state: FrameState
    ):
        """
        Adds new keypoints to the tracks dictionary if the current frame
        is one of the start frames. This is done by comparing the current
        keypoints with the keypoints of the start frame and adding the
        new keypoints to the tracks dictionary.
        """

        if state.t in self.start_frame_indices:
            return

        current_keypoints = state.keypoints
        possible_candidates = np.array([kp.pt for kp in state.features.keypoints])

        mask_x = self.ismembertol(possible_candidates[:, 0], current_keypoints[0, :])
        mask_y = self.ismembertol(possible_candidates[:, 1], current_keypoints[1, :])
        mask = np.logical_or(mask_x, mask_y)

        self.tracks[state.t] = [possible_candidates[mask, :]]
        self.start_frame_indices.append(state.t)

    def track_keypoints(
        self, 
        img_i : np.ndarray, 
        img_j : np.ndarray
    ):
        for start_frame_index in self.start_frame_indices:
            if len(self.tracks[start_frame_index][0]) == 0:
                self.tracks.pop(start_frame_index)
                self.start_frame_indices.remove(start_frame_index)
                continue

            kp_j, mask = klt(self.tracks[start_frame_index][-1], img_i, img_j)
            mask_indices = np.where(mask.reshape(-1, 1) == True)[0]
            kp_j = kp_j[mask_indices].squeeze()
            if len(kp_j.shape) == 1:
                kp_j = kp_j.reshape(1, -1)

            # select only the keypoints based on the good ones of current
            # kp_j tracking
            self.tracks[start_frame_index] = [track[mask_indices] for track in self.tracks[start_frame_index]]
            self.tracks[start_frame_index].append(kp_j)

    def check_for_new_landmarks(
            self, 
            next_frame: int, 
            frame_states: List[FrameState], 
            K: np.ndarray
        ) -> bool:

        for start_frame_index in self.start_frame_indices:
            if next_frame == start_frame_index:
                continue

            track = self.tracks[start_frame_index]

            cam_start_to_world = frame_states[start_frame_index].cam_to_world
            cam_end_to_world = frame_states[next_frame].cam_to_world

            M_cam_start_cam_end = np.linalg.inv(cam_start_to_world) @ cam_end_to_world

            homogenous_start_track = np.vstack([track[0].T, np.ones((1, track[0].shape[0]))])
            homogenous_end_track = np.vstack([track[-1].T, np.ones((1, track[-1].shape[0]))])
            
            P_cam_start = linearTriangulation(
                homogenous_start_track, 
                homogenous_end_track, 
                K @ np.eye(3, 4),
                K @ M_cam_start_cam_end[0:3, :]
            )[:3, :]

            #filter points behind camera and far away
            max_distance = 100
            mask_filter = np.logical_and(P_cam_start[2, :] > 0, np.abs(np.linalg.norm(P_cam_start, axis=0)) < max_distance)
            P_cam_start = P_cam_start[:, mask_filter]
            p_end = homogenous_start_track[:, mask_filter]

            P_world = cam_start_to_world @ np.vstack([P_cam_start, np.ones((1, P_cam_start.shape[1]))])
            f_c = P_world[:3, :] - cam_start_to_world[:3, 3].reshape(-1, 1)
            c = P_world[:3, :] - cam_end_to_world[:3, 3].reshape(-1, 1)

            dot_products = np.diag(np.dot(f_c.T, c))
            start_norms = np.linalg.norm(f_c, axis=0)
            end_norms = np.linalg.norm(c, axis=0)
            norm_product = start_norms * end_norms
            cos_angles = dot_products / norm_product
            angles = np.array([np.arccos(cs_angle) for cs_angle in cos_angles])
            mask_angle = np.where(angles.reshape(-1, 1) > self.angle_threshold)[0]
            
            # TODO: Check if there are landmarks that are similar to some of P_world with np.isclose
            ext_landmarks = np.hstack([frame_states[next_frame].landmarks, P_world[:3, :]])
            frame_states[next_frame].landmarks = ext_landmarks

            # updating the keypoints in each state since there have been multiple
            # filtering steps that reduced the number of initial keypoints
            ext_keypoints_end = np.hstack([frame_states[next_frame].keypoints, p_end[:2, :]])
            frame_states[next_frame].keypoints = ext_keypoints_end

            # remove used keypoints in track since it is now used as landmarks
            self.tracks[start_frame_index] = [track[mask_angle] for track in self.tracks[start_frame_index]]
        return frame_states
    
    def ismembertol(self, arr1, arr2, pixel_distance_threshold=1):
        arr1 = np.atleast_1d(arr1)
        arr2 = np.atleast_1d(arr2)

        # Calculate the absolute differences and check if within tolerance
        diffs = np.abs(arr1[:, None] - arr2)
        return np.any(diffs <= pixel_distance_threshold, axis=1)

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
