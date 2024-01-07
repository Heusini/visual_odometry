from typing import List, Mapping

import cv2 as cv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from features import detect_features
from klt import klt
from state import FrameState
from utils.utils import hom_inv
from dashboard import Dashboard
from pnp import pnp
from utils.dataloader import DataLoader, Dataset
from features import detect_features, FeatureDetector, match_features, matching_klt
from twoDtwoD import twoDtwoD, initialize_camera_poses 
from params import get_params, which_dataset



class Track:
    start_t: int
    end_t: int
    keypoints_start: np.ndarray  # [N , 2] where N is the number of keypoints
    keypoints_end: np.ndarray  # [N , 2] where N is the number of keypoints

    def __init__(self, start_t: int, start_keypoints: np.ndarray) -> None:
        self.start_t = start_t
        self.keypoints_start = start_keypoints
        self.keypoints_end = start_keypoints
        self.end_t = start_t

    def update(self, img_i: np.ndarray, img_j: np.ndarray):
        keypoints_next, klt_mask = klt(
            self.keypoints_end[:, :], 
            img_i, 
            img_j
            )
                
        print(
            f"KLT: tracking {np.sum(klt_mask)} keypoints from track starting at"
            f" {self.start_t}"
        )

        mask = klt_mask

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
    active_tracks: Mapping[int, Track]  # maps from start frame index to track
    inactive_tracks: Mapping[int, Track]  # maps from start frame index to track
    same_keypoints_threshold: float
    max_track_length: int

    def __init__(
        self,
        same_keypoints_threshold: float,
        max_track_length: int,
        max_depth_distance: float,
        init_keyframe: FrameState
    ) -> None:
        self.active_tracks = {}
        self.same_keypoints_threshold = same_keypoints_threshold
        self.max_track_length = max_track_length
        self.max_depth_distance = max_depth_distance
        self.prev_keyframe = init_keyframe

    def start_new_track(self, state: FrameState, check_keypoints: bool = True):
        state.features = detect_features(cv.imread(state.img_path, cv.IMREAD_GRAYSCALE))
        new_kps = state.features.get_positions()
        if check_keypoints:
            landmarks_kps = state.keypoints
            mask_x = self.threshold_mask(new_kps[:, 0], landmarks_kps[:, 0])
            mask_y = self.threshold_mask(new_kps[:, 1], landmarks_kps[:, 1])

            mask = np.logical_not(np.logical_and(mask_x, mask_y))

            print(
                f"filtering out {np.sum(np.logical_not(mask))} keypoints that are too"
                " close to existing landmarks"
            )

            self.active_tracks[state.t] = Track(state.t, new_kps[mask, :])
        else:
            self.active_tracks[state.t] = Track(state.t, new_kps)

    def threshold_mask(self, arr1, arr2):
        arr1 = np.atleast_1d(arr1)
        arr2 = np.atleast_1d(arr2)

        # Calculate the absolute differences and check if within tolerance
        diffs = np.abs(arr1[:, None] - arr2)
        return np.any(diffs <= self.same_keypoints_threshold, axis=1)

    def update(self, t: int, img_i: np.ndarray, img_j: np.ndarray):
        for track_start_t in list(self.active_tracks.keys()):
            # remove tracks that are too long or have no keypoints left
            if (
                self.active_tracks[track_start_t].length() > self.max_track_length
                or 
                self.active_tracks[track_start_t].size() == 0
            ):
                del self.active_tracks[track_start_t]
                continue

            self.active_tracks[track_start_t].update(img_i, img_j)

    def get_new_landmarks(
        self,
        time_j: int,
        min_track_length: int,
        frame_states: List[FrameState],
        K: np.ndarray,
    ) -> (np.ndarray, np.ndarray):
        params = get_params()
        landmarks = np.zeros((0, 3))
        keypoints = np.zeros((0, 2))

        for time_i in list(self.active_tracks.keys()):
            # skip tracks that are too short
            if self.active_tracks[time_i].length() < min_track_length:
                continue
            
            state_i = frame_states[time_i]
            state_j = frame_states[time_j]
            
            if state_i.landmarks.shape[0] > params.CONT_PARAMS.MIN_NUM_LANDMARKS:
                baseline_sigma = self.baseline_uncertainty(state_i.cam_to_world, state_j.cam_to_world, state_i.landmarks)
                if baseline_sigma < params.CONT_PARAMS.BASELINE_SIGMA:
                    continue

            track = self.active_tracks[time_i]

            kp_i = np.hstack([track.keypoints_start, np.ones((track.size(), 1))])
            kp_j = np.hstack([track.keypoints_end, np.ones((track.size(), 1))])
            P_world = cv.triangulatePoints(
                K @ np.linalg.inv(state_i.cam_to_world)[0:3, :],
                K @ np.linalg.inv(state_j.cam_to_world)[0:3, :],
                kp_i[:, :2].T, kp_j[:, :2].T,
            ).T
            P_world = P_world / np.reshape(P_world[:, 3], (-1, 1))

            M_cami_to_camj = np.linalg.inv(state_j.cam_to_world) @ state_i.cam_to_world

            P_camj = (np.linalg.inv(state_j.cam_to_world) @ P_world.T).T
            P_cami = (np.linalg.inv(state_i.cam_to_world) @ P_world.T).T
            P_camj_proj = (M_cami_to_camj @ P_cami.T).T
            error = np.linalg.norm(P_camj[:, :2] - P_camj_proj[:, :2], axis=1)
            # remove points that do not correspond to motion of the camera
            mask_motion = error < np.percentile(error, 80)
            # P_world = P_world[mask_motion, :]
            # kp_j = kp_j[mask_motion, :]

            P_camj = (np.linalg.inv(state_j.cam_to_world) @ P_world.T).T

            print(f"Found {P_world.shape} new landmarks")

            # filter points behind camera and far away
            mask_distance = np.logical_and(
                P_camj[:, 2] > 0, np.abs(np.linalg.norm(P_camj, axis=1)) < self.max_depth_distance
            )
            P_world = P_world[mask_distance, :]
            kp_j = kp_j[mask_distance, :]

            # #remove landmarks where the angle between the two cameras is too small
            T_cami = state_i.cam_to_world[:3, 3]
            T_camj = state_j.cam_to_world[:3, 3]

            P_to_cami = T_cami - P_world[:, :3]
            P_to_camj = T_camj - P_world[:, :3]

            # normalize
            P_to_cami = P_to_cami / np.linalg.norm(P_to_cami, axis=1)[:, None]
            P_to_camj = P_to_camj / np.linalg.norm(P_to_camj, axis=1)[:, None]

            print(P_to_cami[:10, :])
            print(P_to_camj[:10, :])
            # print(P_to_cami.shape)  
            # print(P_to_camj.shape)
            inner = np.sum(P_to_cami*P_to_camj, axis=1)
            print(inner[:10])
            angles = np.arccos(np.clip(inner, -1.0, 1.0))
            print(angles[:10])
            angle_mask = angles > np.percentile(angles, 90)

            print(
                f"filtering out {np.sum(np.logical_not(angle_mask))} landmarks that"
                " have too small angle between cameras"
            )

            # P_world = P_world[angle_mask, :]
            # kp_j = kp_j[angle_mask, :]

            landmarks = np.concatenate([landmarks, P_world[:, :3]], axis=0)
            keypoints = np.concatenate([keypoints, kp_j[:, :2]], axis=0)

            del self.active_tracks[time_i]

        return landmarks, keypoints

    

    def baseline_uncertainty(self, T0: np.ndarray, T1: np.ndarray,
                              landmarks: np.ndarray) -> float:
        T0_inv = hom_inv(T0)
        T1_inv = hom_inv(T1)

        # depth of the landmarks in first camera
        camera_normal = T0[0:3, 0:3].T @ np.array([[0], [0], [1]])
        camera_origin = T0_inv @ np.array([[0], [0], [0], [1]])
        centered_landmarks = landmarks - camera_origin[0:3].ravel()
        depths = []
        for landmark in centered_landmarks:
            d = np.dot(landmark, camera_normal.ravel())
            if d > 0:
                depths.append(d)

        if len(depths) == 0:
            return np.inf
        depth = np.mean(depths)
        # distance of the two poses
        init = T0_inv[:3, 3]
        final = T1_inv[:3, 3]
        dist = np.linalg.norm(final - init)
        return float(dist / depth)

    def plot_stats(self, current_state: FrameState):
        if len(list(self.tracks.keys())) > 0:
            # first plot
            self.num_keypoints = [
                self.tracks[key][0].shape[0] for key in list(self.tracks.keys())
            ]
            keys = list(self.tracks.keys())
            colors = cm.viridis(np.linspace(0, 1, len(keys)))
            self.ax.clear()
            self.ax.bar(keys, self.num_keypoints, color=colors)
            self.ax.set_xlabel("Keys")
            self.ax.set_ylabel("List Lengths")
            self.ax.set_title("List Lengths in Dictionary Over Time")

            # second plot
            self.ax1.plot(current_state.t, current_state.landmarks.shape[1], "o-")
            self.ax1.set_xlabel("frame k")
            self.ax1.set_ylabel("Number of landmarks at k")
            self.ax1.set_title("Tracking the number of landmarks")

            plt.tight_layout()
            plt.draw()
            plt.pause(0.5)

def pairwise_distances(points):
    # Assuming points is an Nx3 numpy array
    dists = np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :]) ** 2, axis=-1))
    return dists

def compute_scale_factor(dists1, dists2):
    # Avoid division by zero
    valid_indices = dists2 != 0
    ratios = dists1[valid_indices] / dists2[valid_indices]
    scale_factor = np.median(ratios)
    return scale_factor



def run_tracking(dataset=Dataset.PARKING, plotting=False, plotting2=False, auto=True):
    # update plots automatically or by clicking
    AUTO = auto
    PLOTTING = plotting
    PLOTTING2 = plotting2
    # select dataset
    which_dataset(dataset.value)
    params = get_params()
    # load data
    # if you remove steps then all the images are used
    loader = DataLoader(dataset, start=0, stride=1, steps=float('inf'))
    print("Loading data...")

    config = loader.config

    loader.load_data()
    print("Data loaded!")

    K, poses, states = loader.get_data()
    print("Data retrieved!")

    ref_frame = params.INIT_PARAMS.BASELINE_FRAME_INDICES[0]
    start_frame = params.INIT_PARAMS.BASELINE_FRAME_INDICES[1]

    max_landmarks_plot = 0
    landmarks_history_count = []


    # Initialize 3D pointcloud
    M_cam_to_world, landmarks, keypoints_i, keypoints_j = twoDtwoD(
        states[ref_frame],
        states[start_frame],
        [s.img_path for s in states[ref_frame:start_frame+1]],
        K,
        feature_detector=params.INIT_PARAMS.MATCHER,
    )

    states[ref_frame].cam_to_world = np.eye(4)
    states[ref_frame].landmarks = landmarks
    states[ref_frame].keypoints = keypoints_i

    states[start_frame].cam_to_world = M_cam_to_world
    states[start_frame].landmarks = landmarks
    states[start_frame].keypoints = keypoints_j

    # init tracking 
    track_manager = TrackManager(
        same_keypoints_threshold=params.CONT_PARAMS.SAME_KEYPOINTS_THRESHOLD,
        max_track_length=params.CONT_PARAMS.MAX_TRACK_LENGTH,
        max_depth_distance=params.CONT_PARAMS.MAX_DEPTH_DISTANCE,
        init_keyframe=states[ref_frame]
    )

    if PLOTTING:
        plt.ion()
        dashboard = Dashboard()
        plt.show()
        plt.pause(0.1)

        cam_hist = np.zeros((len(states), 3))
        cam_hist[start_frame, :] = states[start_frame].cam_to_world[:3, 3]
        cam_hist[ref_frame, :] = states[ref_frame].cam_to_world[:3, 3]
        dashboard.update_axis_with_clear(3, [cam_hist[:, 0], cam_hist[:, 2]], "black")
    elif PLOTTING2 and (dataset == Dataset.PARKING or dataset == Dataset.KITTI):
        plt.ion()
        fig = plt.figure()
        plt.show()
        plt.pause(0.1)
        cam_hist = np.zeros((len(states), 3))
        cam_hist[start_frame, :] = states[start_frame].cam_to_world[:3, 3]
        cam_hist[ref_frame, :] = states[ref_frame].cam_to_world[:3, 3]
        ground_truth = loader.poses
        print(f"{ref_frame} ref_frame")
        rescaled_ground_truth = np.zeros((len(states), 3))
        print(f"rescaled_ground_truth: {len(rescaled_ground_truth)}")


    track_manager.prev_keyframe = states[start_frame]

    for t in range(ref_frame, len(states)):
        state_i = states[t]
        state_j = states[t + 1]

        img_i = cv.imread(state_i.img_path, cv.IMREAD_GRAYSCALE)
        img_j = cv.imread(state_j.img_path, cv.IMREAD_GRAYSCALE)

        img_size = img_i.shape

        print(state_i.keypoints.shape)

        state_j.keypoints, mask_klt = klt(
            state_i.keypoints,
            img_i,
            img_j,
        )

        state_j.keypoints = state_j.keypoints[mask_klt, :]
        state_j.landmarks = state_i.landmarks[mask_klt, :]

        # theoretically: we don't need more than 51 iterations of RANSAC
        # if we assume that sift has 20 % outliers (from lecture) But I add a little margin,
        # that's why I set it to 100
        M_cam_to_world, ransac_mask = pnp(
            state_j.keypoints, 
            state_j.landmarks, 
            K
        )
        ransac_mask = np.squeeze(ransac_mask)

        state_j.cam_to_world = M_cam_to_world
        state_j.landmarks = state_j.landmarks[ransac_mask, :]
        state_j.keypoints = state_j.keypoints[ransac_mask, :]

        P_world = np.hstack([state_j.landmarks, np.ones((state_j.landmarks.shape[0], 1))])

        P_camj = (np.linalg.inv(state_j.cam_to_world) @ P_world.T).T

        # filter points behind camera and far away
        mask_distance = np.logical_and(
            P_camj[:, 2] > 0, np.abs(np.linalg.norm(P_camj, axis=1)) < 100
        )
        state_j.landmarks = state_j.landmarks[mask_distance, :]
        state_j.keypoints = state_j.keypoints[mask_distance, :]

        track_manager.update(
            t,
            img_i=img_i,
            img_j=img_j,
        )

        track_manager.start_new_track(state_j, check_keypoints=True)

        landmarks, keypoints = track_manager.get_new_landmarks(
            t + 1,
            min_track_length=params.CONT_PARAMS.MIN_TRACK_LENGTH,
            frame_states=states,
            K=K,
        )
        
        print(f"Found {landmarks.shape} new landmarks")


        if PLOTTING:
            if t % 30 == 0:
                dashboard.clear_axis(2)
                dashboard.update_axis_description(2, "all landmarks", "x", "z")
            
            img = img_j

            print(f"{state_j.keypoints.shape} keypoints shape")
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
            dashboard.update_image(0, img, [state_j.keypoints, keypoints], colors=colors)

            landmarks_camera_space = np.hstack([state_j.landmarks, np.ones((state_j.landmarks.shape[0], 1))])
            landmarks_camera_space = (np.linalg.inv(state_j.cam_to_world) @ landmarks_camera_space.T ).T

            limit_points = []
            limit_points.append([landmarks_camera_space[:, 0], landmarks_camera_space[:, 2]])

            
            dashboard.update_axis_with_clear(1, [landmarks_camera_space[:, 0], landmarks_camera_space[:, 2]], color='orange', label="previous_landmarks")
            dashboard.update_axis_description(1, "landmarks", "x", "z")
            if landmarks.shape[0] > 0:
                landmarks_new_camera_space = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])
                landmarks_new_camera_space = (np.linalg.inv(state_j.cam_to_world) @ landmarks_new_camera_space.T ).T

                dashboard.update_axis(1, [landmarks_new_camera_space[:, 0], landmarks_new_camera_space[:, 2]], color='green', label="new landmarks")
                limit_points.append([landmarks_new_camera_space[:, 0], landmarks_new_camera_space[:, 2]])

            limit_points.append([[0], [0]])
            # calculate limits
            limits = dashboard.calculate_limits_from_points(limit_points)
            # set limits
            dashboard.set_limits(1, limits[0], limits[1])
            dashboard.update_axis(1, [[0],[0]], color='red', label="camera position")

            x_cam = np.array([[0.2, 0, 0, 1], [-0.2, 0, 0, 1]])
            T_cam = (state_j.cam_to_world @ x_cam.T).T
            
            cam_hist[t+1, :] = (state_j.cam_to_world[:3, 3])

            label = "all landmarks" if t == 0 else None
            cam_label = "camera position" if t == 0 else None
            dashboard.update_axis(2, [state_j.landmarks[:, 0], state_j.landmarks[:, 2]], color='black', label=label)
            dashboard.update_axis(2, [state_j.cam_to_world[0, 3], state_j.cam_to_world[2, 3]], color='red', label=cam_label)

            dashboard.update_axis(3, [cam_hist[ref_frame:t+1, 0], cam_hist[ref_frame:t+1, 2]], color='black', label=cam_label)
            limits = dashboard.calculate_limits_from_points([[cam_hist[ref_frame:t+1, 0], cam_hist[ref_frame:t+1, 2]]])
            dashboard.set_limits(3, limits[0], limits[1])

            label = "landmarks over time" if t == 0 else None

            ### add history of landmark count
            print(f"landmarks shape {state_j.landmarks.shape}")
            landmarks_history_count.append([t, state_j.landmarks.shape[0]])
            dashboard.update_axis_line(4, np.asarray(landmarks_history_count), color='red', label=label)
            max_landmarks_plot = max_landmarks_plot if state_j.landmarks.shape[0] < max_landmarks_plot else state_j.landmarks.shape[0]
            dashboard.set_limits(4, [0, t+5], [0, max_landmarks_plot+20])

            plt.draw()
            if AUTO:
                plt.pause(0.05)
            else:
                plt.waitforbuttonpress()
            if t == 0:
                dashboard.update_axis_description(2, "all landmarks", "x", "z")
                dashboard.update_axis_description(3, "camera position over time", "x", "z")
        elif PLOTTING2 and (dataset == Dataset.KITTI or dataset == Dataset.PARKING) and t > 1:
            cam_hist[t+1, :] = (state_j.cam_to_world[:3, 3])
            vec_norm = np.linalg.norm(cam_hist[t+1,:] - cam_hist[t,:])
            scale_factor = vec_norm / np.linalg.norm(ground_truth[t+1, :3, 3] - ground_truth[t, :3, 3])


            rescaled_vec = scale_factor*(ground_truth[t+1, :3, 3] - ground_truth[t, :3, 3])
            rescaled_ground_truth[t+1] = rescaled_vec + rescaled_ground_truth[t]
            plt.plot(cam_hist[t, 0], cam_hist[t, 2],"x", color='blue')

            plt.plot(rescaled_ground_truth[t][0], rescaled_ground_truth[t][2], "o", color="red")
            plt.draw()
            plt.pause(0.05)


        # update state
        state_j.keypoints = np.concatenate([state_j.keypoints, keypoints], axis=0)
        state_j.landmarks = np.concatenate([state_j.landmarks, landmarks], axis=0)

        # randomize order of landmarks and keypoints
        # indices = np.arange(state_j.keypoints.shape[0])
        # np.random.shuffle(indices)
        # state_j.keypoints = state_j.keypoints[indices, :]
        # state_j.landmarks = state_j.landmarks[indices, :]

if __name__ == "__main__":
    run_tracking(dataset=Dataset.KITTI, plotting2=True)




