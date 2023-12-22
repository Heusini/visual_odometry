from typing import NamedTuple, List
import numpy as np
from camera import Camera
from feature_detection import Keypoints
import cv2 as cv

class Transform3D(NamedTuple):
    rotation : np.ndarray
    translation : np.ndarray

    def to_homogeneous_matrix(self):
        M = np.eye(4)
        M[:3, :3] = self.rotation
        M[:3, 3:] = self.translation
        return M

class FrameSate(NamedTuple):
    t : int
    img_path : str
    world_to_cam : Transform3D
    keypoints : List[Keypoints] # $P$ from proj. statement
    keypoints_descriptors : np.ndarray # $P$ from proj. statement
    landmarks : np.ndarray # $X$ from proj. statement
