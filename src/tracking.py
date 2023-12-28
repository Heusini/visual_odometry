import cv2 as cv
import numpy as np
from state import FrameState
from utils.linear_triangulation import linearTriangulation
from klt import klt
from typing import List

import matplotlib.pyplot as plt
import matplotlib.cm as cm
# TODO: I propose this archtecture instead. I we can simplify the code a lot with this.
# instead of using the tracking class, we would then simply maintain a list of tracks in the main loop
class Track:
    start_t : int
    keypoints : np.ndarray # [L, N , 2] where L is the number of frames and N is the number of keypoints

    def __init__(self, start_t: int, keypoints: List[np.ndarray]) -> None:
        self.start_t = start_t
        self.keypoints = keypoints

    def track(self, img_i: np.ndarray, img_j: np.ndarray):
        kp_j, mask = klt(self.keypoints[-1, : , :], img_i, img_j)
        self.keypoints = self.keypoints[:, mask]
        self.keypoints = np.vstack([self.keypoints, kp_j])

def compute_new_landmarks(
    tracks : List[Track],
    min_length : int,
) -> np.ndarray:
    """
    Searches for new landmarks in the tracks and returns them as a numpy array.
    """
    pass

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
