from typing import Optional
import numpy as np
from features import Features
import cv2 as cv
from utils.geometry import Transform3D
from features import detect_features

class FrameSate():
    t : int
    img_path : str
    cam_to_world : np.ndarray
    features : Features
    landmarks : np.ndarray # $X$ from proj. statement

    def __init__(
        self,
        t : int,
        img_path : str,
        cam_to_world : Optional[np.ndarray] = None,
        features : Optional[Features] = None,
        landmarks : Optional[np.ndarray] = None
    ):
        self.t = t
        self.img_path = img_path
        self.cam_to_world = cam_to_world if cam_to_world is not None else np.eye(4)
        self.features = features if features else detect_features(cv.imread(img_path))
        self.landmarks = landmarks