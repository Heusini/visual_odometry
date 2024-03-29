from typing import Optional

import cv2 as cv
import numpy as np

from features import Features, detect_features


class FrameState:
    t: int  # frame index (time)
    img_path: str
    cam_to_world: np.ndarray
    keypoints: np.ndarray  # [N, 2] array of 2D keypoints
    kp_track_start: np.ndarray  # [N] array of keypoint track start frame indices
    features: Features
    landmarks: np.ndarray  # $X$ from proj. statement

    def __init__(
        self,
        t: int,
        img_path: str,
        cam_to_world: Optional[np.ndarray] = None,
        features: Optional[Features] = None,
        landmarks: Optional[np.ndarray] = None,
    ):
        self.t = t
        self.img_path = img_path
        self.cam_to_world = cam_to_world if cam_to_world is not None else np.eye(4)
        self.features = features
        self.landmarks = landmarks
