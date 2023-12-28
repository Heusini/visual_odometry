import cv2 as cv
import numpy as np
from state import FrameState
from utils.linear_triangulation import linearTriangulation
from klt import klt
from typing import List

class Tracking:
    def __init__(self, angle_threshold: float, init_frame_indices: [int, int]) -> None:
        self.tracks = {}
        self.start_frame_indices = []
        self.angle_threshold = angle_threshold
        self.init_frame_indices = init_frame_indices
        pass
    
    def add_keypoint_candidates(self, keypoint_candidates: np.ndarray, start_frame_index: int):
        if start_frame_index not in self.init_frame_indices:
            self.tracks[start_frame_index] = [np.array([kp.pt for kp in keypoint_candidates])]
            self.start_frame_indices.append(start_frame_index)
        else:
            pass

    def track_keypoints(self, img_i : np.ndarray, img_j : np.ndarray):
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

    def check_for_new_landmarks(self, frame_states: List[FrameState], K: np.ndarray) -> bool:
        for start_frame_index in self.start_frame_indices:
            track = self.tracks[start_frame_index]
            end_frame_index = start_frame_index + len(track) - 1

            cam_start_to_world = frame_states[start_frame_index].cam_to_world
            cam_end_to_world = frame_states[end_frame_index].cam_to_world

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
            p_start = homogenous_start_track[:, mask_filter]
            p_end = homogenous_start_track[:, mask_filter]

            P_cam_end = (P_cam_start[:3, :].T - M_cam_start_cam_end[:3, 3]).T

            dot_products = np.diag(np.dot(P_cam_start.T, P_cam_end))
            start_norms = np.linalg.norm(P_cam_start, axis=0)
            end_norms = np.linalg.norm(P_cam_end, axis=0)
            norm_product = start_norms * end_norms
            cos_angles = dot_products / norm_product
            angles = np.array([np.arccos(cs_angle) for cs_angle in cos_angles])
            mask_angle = np.where(angles.reshape(-1, 1) > self.angle_threshold)[0]
            
            P_world = cam_start_to_world @ np.vstack([P_cam_start, np.ones((1, P_cam_start.shape[1]))])

            # TODO: Check if there are landmarks that are similar to some of P_world with np.isclose
            ext_landmarks = np.hstack([frame_states[start_frame_index].landmarks, P_world[:3, :]])
            frame_states[start_frame_index].landmarks = ext_landmarks
            frame_states[end_frame_index].landmarks = ext_landmarks

            # updating the keypoints in each state since there have been multiple
            # filtering steps that reduced the number of initial keypoints
            ext_keypoints_start = np.hstack([frame_states[start_frame_index].keypoints, p_start[:2, :]])
            ext_keypoints_end = np.hstack([frame_states[start_frame_index].keypoints, p_end[:2, :]])
            frame_states[start_frame_index].keypoints = ext_keypoints_start
            frame_states[end_frame_index].keypoints = ext_keypoints_end

            # remove track since it is now used as landmarks
            self.tracks[start_frame_index] = [track[mask_angle] for track in self.tracks[start_frame_index]]

        return frame_states