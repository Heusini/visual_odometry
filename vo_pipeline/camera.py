import cv2 as cv
import numpy as np
from correspondence import correspondence
from exercise_helpers.decompose_essential_matrix import decomposeEssentialMatrix
from exercise_helpers.disambiguate_relative_pose import disambiguateRelativePose
from exercise_helpers.linear_triangulation import linearTriangulation
from feature_detection import feature_detection, feature_matching


def get_fundamental_matrix(keypoints_a, keypoints_b):
    fundamental_mat, mask = cv.findFundamentalMat(
        keypoints_a, keypoints_b, cv.RANSAC, 1, 0.9999, 50000
    )
    return fundamental_mat, mask


def get_essential_matrix(point_a, point_b, K):
    essential_mat, mask_e = cv.findEssentialMat(
        point_a, point_b, K, cv.RANSAC, 0.9999, 1
    )
    return essential_mat, mask_e


def essential_matrix_from_fundamental_matrix(fundamental_mat, K):
    return K.T @ fundamental_mat @ K


class Camera:
    def __init__(
        self,
        id: int,
        K: np.ndarray,
        image,
        rotation=np.eye(3),
        translation=np.zeros((3, 1)),
    ):
        """
        Parameters
        ----------
        id: id to identify cameras should be unique
        K: np.ndarry of shape (3, 3) intrinsic of camera
        rotation: np.ndarray of shape (3, 3) rotation matrix of R_W_C
        translation: np.ndarray of shape (3, 1) translation vector of c2T_C_W

        Returns
        -------
        Camera object
        """
        self.id = id
        self.rotation = rotation
        self.translation = translation
        self.features = None
        self.matches = None
        self.point_cloud = None
        self.image = image
        self.K = K

    def calculate_features(self):
        self.features = feature_detection(self.image)

    def calculate_matches(self, other: "Camera", threshold=0.7):
        return correspondence(self.features, other.features, threshold)

    def calculate_cam2_pose(
        self, cam2: "Camera", matched_points_cam1, matched_points_cam2
    ):
        """
        Parameters
        ----------
        cam2: Camera object of second camera
        matched_points_cam1: points that matched between cam1 and cam2
        matched_points_cam2: points that matched between cam1 and cam2

        Returns
        -------
        mask_f: mask of inliers of fundamental matrix for filtering points
        """
        F, mask_f = get_fundamental_matrix(matched_points_cam1, matched_points_cam2)
        E = essential_matrix_from_fundamental_matrix(F, self.K)
        p1 = np.hstack(
            [matched_points_cam1, np.ones((matched_points_cam1.shape[0], 1))]
        ).T
        p2 = np.hstack(
            [matched_points_cam2, np.ones((matched_points_cam2.shape[0], 1))]
        ).T
        p1 = p1[:, mask_f.ravel() == 1]
        p2 = p2[:, mask_f.ravel() == 1]
        Rots, u3 = decomposeEssentialMatrix(E)
        R_CAM2_CAM1, T_CAM2_CAM1 = disambiguateRelativePose(
            Rots, u3, p1, p2, self.K, self.K
        )

        M1 = self.K @ np.hstack([self.rotation, self.translation])
        M2 = cam2.K @ np.hstack([cam2.rotation, cam2.translation])
        points_3d = cv.triangulatePoints(
            M1, M2, matched_points_cam1.T, matched_points_cam2.T
        )

        # TODO: check if this is correct
        # self.rotation is of R_W_C
        cam2.rotation = self.rotation @ R_CAM2_CAM1.T
        # keine ahnung ob das richtig ist
        cam2.translation = (
            -cam2.rotation @ T_CAM2_CAM1 - self.rotation @ self.translation
        )
        return mask_f

    def create_point_cloud(
        self, cam2: "Camera", matched_points_cam1, matched_points_cam2, mask_f
    ):
        pass
