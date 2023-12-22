from typing import Optional
import numpy as np
from features import Features
import cv2 as cv
from utils.geometry import Transform3D
from features import detect_features

class FrameSate():
    t : int
    img_path : str
    world_to_cam : Transform3D
    features : Features
    landmarks : np.ndarray # $X$ from proj. statement

    def __init__(
        self,
        t : int,
        img_path : str,
        world_to_cam : Optional[Transform3D] = None,
        features : Optional[Features] = None,
        landmarks : Optional[np.ndarray] = None
    ):
        self.t = t
        self.img_path = img_path
        self.world_to_cam = world_to_cam if world_to_cam is not None else Transform3D()
        self.features = features if features else detect_features(cv.imread(img_path))
        self.landmarks = landmarks