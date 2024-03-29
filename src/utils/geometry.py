from typing import Optional

import cv2
import numpy as np
from params import get_params

class Transform3D:
    def __init__(self, R: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None):
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros((3, 1))

    def to_mat(self):
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3:] = self.t
        return M

    def get_R(self):
        M = np.eye(4)
        M[:3, :3] = self.R
        return M

    def get_t(self):
        M = np.eye(4)
        M[:3, 3:] = self.t
        return M

    # rotate and then translate
    def Rt(self):
        return self.get_t() @ self.get_R()

    # translate and then rotate
    def tR(self):
        return self.get_R() @ self.get_t()


def calc_fundamental_mat(points1, points2):
    params = get_params()
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.RANSAC, params.INIT_PARAMS.RANSAC_PARAMS_F.THRESHOLD, params.INIT_PARAMS.RANSAC_PARAMS_F.CONFIDENCE, params.INIT_PARAMS.RANSAC_PARAMS_F.NUM_ITERATIONS)
    return F, mask


def calc_essential_mat(points1, points2, K, reprojection_error=1, confidence=0.9999, num_iterations=50000):
    E, mask_e = cv2.findEssentialMat(points1, points2, K, cv2.RANSAC, confidence, reprojection_error, num_iterations)
    return E, mask_e


def calc_essential_mat_from_fundamental_mat(F, K):
    return K.T @ F @ K
