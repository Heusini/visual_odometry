from typing import List, Mapping

import cv2 as cv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from features import detect_features
from klt import klt
from state import FrameState
from utils.utils import hom_inv

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

    def update(self, img_i: np.ndarray, img_j: np.ndarray, lk_params):
        keypoints_next, klt_mask = klt(
            self.keypoints_end[:, :], 
            img_i, 
            img_j,
            lk_params=lk_params)
        
        img_size = img_i.shape
        
        print(
            f"KLT: tracking {np.sum(klt_mask)} keypoints from track starting at"
            f" {self.start_t}"
        )

        # min_border_distance = 0.2 * np.min(img_size)
        # mask_x = np.logical_and(
        #     keypoints_next[:, 0] > min_border_distance,
        #     keypoints_next[:, 0] < img_size[1] - min_border_distance,
        # )
        # mask_y = np.logical_and(
        #     keypoints_next[:, 1] > min_border_distance,
        #     keypoints_next[:, 1] < img_size[0] - min_border_distance,
        # )

        # border_mask = np.logical_and(mask_x, mask_y)

        # mask = np.logical_and(klt_mask, border_mask)
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
    angle_threshold: float
    same_keypoints_threshold: float
    max_track_length: int

    def __init__(
        self,
        angle_threshold: float,
        same_keypoints_threshold: float,
        max_track_length: int,
        max_depth_distance: float,
        init_keyframe: FrameState
    ) -> None:
        self.active_tracks = {}
        self.angle_threshold = angle_threshold
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

    def update(self, t: int, img_i: np.ndarray, img_j: np.ndarray, lk_params: dict):
        for track_start_t in list(self.active_tracks.keys()):
            # remove tracks that are too long or have no keypoints left
            if (
                self.active_tracks[track_start_t].length() > self.max_track_length
                or self.active_tracks[track_start_t].size() == 0
            ):
                del self.active_tracks[track_start_t]
                continue

            self.active_tracks[track_start_t].update(img_i, img_j, lk_params=lk_params)

    def get_new_landmarks(
        self,
        time_j: int,
        min_track_length: int,
        frame_states: List[FrameState],
        K: np.ndarray,
        compare_to_landmarks: bool = True,
    ) -> (np.ndarray, np.ndarray):
        landmarks = np.zeros((0, 3))
        keypoints = np.zeros((0, 2))

        for time_i in list(self.active_tracks.keys()):
            # skip tracks that are too short
            if self.active_tracks[time_i].length() < min_track_length:
                continue
            state_i = frame_states[time_i]
            state_j = frame_states[time_j]

            # baseline_sigma = self._baseline_uncertainty(self.prev_keyframe.cam_to_world, state_j.cam_to_world, state_j.landmarks)

            # if state_j.landmarks.shape[0] > 5:
            #     if baseline_sigma < 0.15:
            #         continue
        
            # M_cam_to_world, landmarks, keypoints = twoDtwoD(
            #     self.prev_keyframe,
            #     state_j,
            #     [s.img_path for s in frame_states[self.prev_keyframe.t:time_j+1]],
            #     K,
            #     feature_detector=FeatureDetector.SIFT,
            #     ransac_params=(
            #         0.1,
            #         0.999,
            #         10000
            #     ),
            #     lk_params=dict(winSize=(21, 21),maxLevel=8,criteria=(3, 10, 0.001)),
            #     sift_params=0.6,
            #     max_depth_distance=100
            # )
            # self.prev_keyframe = state_j
            # return landmarks, keypoints, M_cam_to_world
            track = self.active_tracks[time_i]

            kp_i = np.hstack([track.keypoints_start, np.ones((track.size(), 1))])
            kp_j = np.hstack([track.keypoints_end, np.ones((track.size(), 1))])
            P_world = cv.triangulatePoints(
                K @ np.linalg.inv(state_i.cam_to_world)[0:3, :],
                K @ np.linalg.inv(state_j.cam_to_world)[0:3, :],
                kp_i[:, :2].T, kp_j[:, :2].T,
            ).T
            P_world = P_world / np.reshape(P_world[:, 3], (-1, 1))

            P_camj = (np.linalg.inv(state_j.cam_to_world) @ P_world.T).T

            print(f"Found {P_world.shape} new landmarks")

            # filter points behind camera and far away
            mask_distance = np.logical_and(
                P_camj[:, 2] > 0, np.abs(np.linalg.norm(P_camj, axis=1)) < self.max_depth_distance
            )
            P_world = P_world[mask_distance, :]
            kp_j = kp_j[mask_distance, :]

            # reprojection = (K @ np.linalg.inv(state_j.cam_to_world)[0:3, :] @ P_world.T).T
            # reprojection /= np.reshape(reprojection[:, 2], (-1, 1))

            # error = np.linalg.norm(reprojection[:, :2] - kp_j[:, :2], axis=1)
            # print(f"mean reprojection error: {np.mean(error)}")

            # error_mask = error < 5
            # P_world = P_world[error_mask, :]
            # kp_j = kp_j[error_mask, :]

            # #remove landmarks where the angle between the two cameras is too small
            # T_cami = state_i.cam_to_world[:3, 3]
            # T_camj = state_j.cam_to_world[:3, 3]

            # P_to_cami = P_world[:, :3] - T_cami
            # P_to_camj = P_world[:, :3] - T_camj
            # dot = np.einsum("ij,ij->i", P_to_cami, P_to_camj)
            # angles = np.arccos(np.clip(dot, -1.0, 1.0))
            # angle_mask = angles > self.angle_threshold

            # print(
            #     f"filtering out {np.sum(np.logical_not(angle_mask))} landmarks that"
            #     " have too small angle between cameras"
            # )

            # P_world = P_world[angle_mask, :]
            # kp_j = kp_j[angle_mask, :]

            # #remove landmarks that are too close to existing landmarks
            # if compare_to_landmarks:
            #     mask_x = self.threshold_mask(P_world[:, 0], state_j.landmarks[:, 0])
            #     mask_y = self.threshold_mask(P_world[:, 1], state_j.landmarks[:, 1])
            #     mask_z = self.threshold_mask(P_world[:, 2], state_j.landmarks[:, 2])

            #     mask_pos = np.logical_not(
            #         np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
            #     )

            #     print(
            #         f"filtering out {np.sum(np.logical_not(mask_pos))} landmarks that"
            #         " are too close to existing landmarks"
            #     )

            #     P_world = P_world[mask_pos, :]
            #     kp_j = kp_j[mask_pos, :]

            landmarks = np.concatenate([landmarks, P_world[:, :3]], axis=0)
            keypoints = np.concatenate([keypoints, kp_j[:, :2]], axis=0)

            del self.active_tracks[time_i]

        return landmarks, keypoints

    

    def _baseline_uncertainty(self, T0: np.ndarray, T1: np.ndarray,
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


if __name__ == "__main__":
    from pnp import pnp
    from utils.dataloader import DataLoader, Dataset
    from features import detect_features, FeatureDetector
    from twoDtwoD import twoDtwoD
    
    # update plots automatically or by clicking
    AUTO = True
    # select dataset
    dataset = Dataset.KITTI

    # load data
    # if you remove steps then all the images are used
    loader = DataLoader(dataset, start=0, stride=1, steps=float('inf'))
    print("Loading data...")

    config = loader.config

    loader.load_data()
    print("Data loaded!")

    K, poses, states = loader.get_data()
    print("Data retrieved!")

    # organize params
    ## Global params
    lk_params = config.klt_params.__dict__
    sift_params = config.sift_params

    ## init params
    ransac_params_F = config.init_params.ransac_params_F
    ref_frame = config.init_params.baseline_frame_indices[0]
    start_frame = config.init_params.baseline_frame_indices[1]
    
    ## cont params
    cont_params = config.continuous_params
    pnp_ransac_params = config.continuous_params.pnp_ransac_params

    ## TODO: plotting params?

    # Initialize 3D pointcloud
    M_cam_to_world, landmarks, keypoints = twoDtwoD(
        states[ref_frame],
        states[start_frame],
        [s.img_path for s in states[ref_frame:start_frame+1]],
        K,
        feature_detector=config.init_params.matcher,
        ransac_params=(
            ransac_params_F.threshold,
            ransac_params_F.confidence,
            ransac_params_F.num_iterations
        ),
        lk_params=lk_params,
        sift_params=sift_params,
        max_depth_distance=config.init_params.max_depth_distance
    )

    print(f"Found {landmarks.shape} new landmarks")
    print(f"Found {keypoints.shape} new keypoints")

    states[ref_frame].cam_to_world = np.eye(4)
    states[ref_frame].landmarks = landmarks
    states[ref_frame].keypoints = keypoints

    states[start_frame].cam_to_world = M_cam_to_world
    states[start_frame].landmarks = landmarks
    states[start_frame].keypoints = keypoints


    # init tracking 
    track_manager = TrackManager(
        angle_threshold=cont_params.angle_threshold,
        same_keypoints_threshold=cont_params.same_keypoints_threshold,
        max_track_length=cont_params.max_track_length,
        max_depth_distance=cont_params.max_depth_distance,
        init_keyframe=states[ref_frame]
    )

    plt.ion()
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])
    plt.show()

    cam_hist = np.zeros((len(states), 3))

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
            lk_params=lk_params
        )

        state_j.keypoints = state_j.keypoints[mask_klt, :]
        state_j.landmarks = state_i.landmarks[mask_klt, :]

        # theoretically: we don't need more than 51 iterations of RANSAC
        # if we assume that sift has 20 % outliers (from lecture) But I add a little margin,
        # that's why I set it to 100
        if state_i.landmarks.shape[0] > 5:
            M_cam_to_world, ransac_mask = pnp(
                state_j.keypoints, 
                state_j.landmarks, 
                K, 
                num_iterations=pnp_ransac_params.num_iterations,
                reprojection_error=pnp_ransac_params.threshold,
                confidence=pnp_ransac_params.confidence
            )
            ransac_mask = np.squeeze(ransac_mask)

            state_j.cam_to_world = M_cam_to_world
            state_j.landmarks = state_j.landmarks[ransac_mask, :]
            state_j.keypoints = state_j.keypoints[ransac_mask, :]
        else:
            M_cam_to_world, landmarks, keypoints = twoDtwoD(
                track_manager.prev_keyframe,
                state_j,
                [s.img_path for s in states[track_manager.prev_keyframe.t:t+1]],
                K,
                feature_detector=FeatureDetector.SIFT,
                ransac_params=(
                    0.1,
                    0.999,
                    10000
                ),
                lk_params=dict(winSize=(21, 21),maxLevel=8,criteria=(3, 10, 0.001)),
                sift_params=0.6,
                max_depth_distance=100
            )

            state_j.cam_to_world = M_cam_to_world
            state_j.landmarks = state_j.landmarks
            state_j.keypoints = state_j.keypoints


        track_manager.update(
            t,
            img_i=img_i,
            img_j=img_j,
            lk_params=lk_params
        )

        track_manager.start_new_track(state_j, check_keypoints=False)

        landmarks, keypoints = track_manager.get_new_landmarks(
            t + 1,
            min_track_length=cont_params.min_track_length,
            frame_states=states,
            K=K,
            compare_to_landmarks=False,
        )

        # baseline_sigma = track_manager._baseline_uncertainty(track_manager.prev_keyframe.cam_to_world, state_j.cam_to_world, state_j.landmarks)

        # if baseline_sigma > 0.1:
        #     M_cam_to_world, landmarks, keypoints = twoDtwoD(
        #         track_manager.prev_keyframe,
        #         state_j,
        #         [s.img_path for s in states[track_manager.prev_keyframe.t:t+1]],
        #         K,
        #         feature_detector=FeatureDetector.SIFT,
        #         ransac_params=(
        #             0.1,
        #             0.999,
        #             10000
        #         ),
        #         lk_params=dict(winSize=(21, 21),maxLevel=8,criteria=(3, 10, 0.001)),
        #         sift_params=0.8,
        #         max_depth_distance=100
        #     )
        # track_manager.prev_keyframe = state_j
        # state_j.cam_to_world = M_cam_to_world
        # state_j.keypoints = keypoints
        # state_j.landmarks = landmarks
        
        print(f"Found {landmarks.shape} new landmarks")

        ax0.clear()
        ax1.clear()

        if t % 30 == 0:
            ax2.clear()
        
        # plot image using cv
        img = cv.cvtColor(img_j, cv.COLOR_BGR2RGB)

        # plot keypoint tracks using cv2
        # n_tracks = len(track_manager.active_tracks)
        # for track_index, track in enumerate(track_manager.active_tracks.values()):
        #     start = track.keypoints_start.astype(int)
        #     end = track.keypoints_end.astype(int)

        #     for i in range(track.size()):
        #         # above line plot code but add to ax[0]
        #         cv.line(
        #             img,
        #             (start[i, 0], start[i, 1]),
        #             (end[i, 0], end[i, 1]),
        #             color=(255, 0, 0),
        #             thickness=1,
        #         )

        # plot keypoints using cv2
        # for i in range(keypoints.shape[0]):
        #     cv.circle(
        #         img,
        #         (int(keypoints[i, 0]), int(keypoints[i, 1])),
        #         radius=3,
        #         color=(0, 0, 255),
        #         thickness=-1,
        #     )

        # plot existing keypoints using cv2
        for i in range(state_j.keypoints.shape[0]):
            cv.circle(
                img,
                (
                    int(state_j.keypoints[i, 0]),
                    int(state_j.keypoints[i, 1]),
                ),
                radius=3,
                color=(0, 255, 0),
                thickness=-1,
            )

        ax0.imshow(img)

        landmarks_camera_space = np.hstack([state_j.landmarks, np.ones((state_j.landmarks.shape[0], 1))])
        landmarks_camera_space = (np.linalg.inv(state_j.cam_to_world) @ landmarks_camera_space.T ).T
        ax1.scatter(
            landmarks_camera_space[:, 0],
            landmarks_camera_space[:, 2],
            s=20,
            c="green",
            label="previous landmarks",
        )
    
        if landmarks.shape[0] > 0:
            landmarks_camera_space = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])
            landmarks_camera_space = (np.linalg.inv(state_j.cam_to_world) @ landmarks_camera_space.T ).T

            ax1.scatter(
                landmarks_camera_space[:, 0],
                landmarks_camera_space[:, 2],
                s=20, 
                c="blue", 
                label="new landmarks"
            )

        ax1.set_xlim(-20, 20)
        ax1.set_ylim(-2, 70)

        x_cam = np.array([[0.2, 0, 0, 1], [-0.2, 0, 0, 1]])
        print(x_cam.shape)
        T_cam = (state_j.cam_to_world @ x_cam.T).T
    
        ax1.scatter(
            0,
            0,
            s=20,
            c="red",
            label="camera position",
        )
        ax1.legend()

        ax1.set_title("landmarks in camera frame")
        ax1.set_xlabel("x")
        ax1.set_ylabel("z")
        #ax1.set_aspect("equal", adjustable="box")

        cam_hist[t, :] = (state_j.cam_to_world[:3, 3])

        ax2.scatter(
            landmarks[:, 0],
            landmarks[:, 2],
            s=1,
            c="black",
            label="all landmarks",
            alpha=0.01,
        )

        ax2.scatter(
            state_j.cam_to_world[0, 3],
            state_j.cam_to_world[2, 3],
            s=3,
            c="red",
            label="camera position",
            # set to cross
            marker="x",

        )
        # add legend
        if t == start_frame:
            ax2.legend()
            ax2.set_title("History of all landmarks in world frame")
            ax2.set_xlabel("x")
            ax2.set_ylabel("z")
            

        #ax2.set_aspect("equal", adjustable="box")
        ax3.clear()
        ax3.scatter(
            cam_hist[:, 0],
            cam_hist[:, 2],
            s=1,
            c="black",
            label="camera position",
            alpha=1,
        )
        min_x = np.min(cam_hist[:, 0])
        max_x = np.max(cam_hist[:, 0])
        margin = 2
        ax3.set_xlim(xmin=min(-2, min_x - margin), xmax=max(2, max_x + margin))
        state_j.keypoints = np.concatenate([state_j.keypoints, keypoints], axis=0)
        state_j.landmarks = np.concatenate([state_j.landmarks, landmarks], axis=0)

        plt.draw()
        if AUTO:
            plt.pause(0.1)
        else:
            plt.waitforbuttonpress()
