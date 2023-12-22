import numpy as np
import cv2
from typing import Optional

class Transform3D():

    def __init__(self, R : Optional[np.ndarray] = None, t : Optional[np.ndarray] = None):
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros((3, 1))
        
    def to_homogeneous_matrix(self):
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3:] = self.t
        return M


def calc_fundamental_mat(points1, points2):
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.RANSAC, 1, 0.9999, 50000)
    return F, mask

def calc_essential_mat(points1, points2, K):
    E, mask_e = cv2.findEssentialMat(points1, points2, K, cv2.RANSAC, 0.9999, 1, 50000)
    return E, mask_e

def calc_essential_mat_from_fundamental_mat(F, K):
    return K.T @ F @ K