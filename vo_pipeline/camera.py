import cv2 as cv
import numpy as np
from correspondence import correspondence
from exercise_helpers.decompose_essential_matrix import decomposeEssentialMatrix
from exercise_helpers.disambiguate_relative_pose import disambiguateRelativePose
from exercise_helpers.linear_triangulation import linearTriangulation
from feature_detection import feature_detection, feature_matching


def get_fundamental_matrix(keypoints_a, keypoints_b):
    # CV2 uses the 8-point algorithm when n > 8
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
    # by definition of the fundamental matrix
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
        rotation: np.ndarray of shape (3, 3) rotation matrix of R_C_W
        translation: np.ndarray of shape (3, 1) translation vector of cT_C_W (small c is Transformation is in the camera frame)

        Returns
        -------
        Camera object
        """
        self.id = id
        # To get the camera point in worldframes we do center = -rotation.T @ translation
        # this rotation is R_CAM_WORLD
        self.rotation = rotation
        # this translation is in CAM
        self.translation = translation

        self.features = None
        self.matches = None
        self.point_cloud = None
        self.image = image
        self.K = K

    def get_center_in_world(self):
        return -self.rotation.T @ self.translation[:, 0]

    def calculate_features(self):
        self.features = feature_detection(self.image)

    def calculate_matches(self, other: "Camera", threshold=0.7):
        return correspondence(self.features, other.features, threshold)

    def calculate_points_in_world(
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
        R_CAM2_CAM1, c2T_CAM2_CAM1 = disambiguateRelativePose(
            Rots, u3, p1, p2, self.K, self.K
        )

        M1 = self.K @ np.eye(3, 4)
        M2 = cam2.K @ np.c_[R_CAM2_CAM1, c2T_CAM2_CAM1]
        P = linearTriangulation(p1, p2, M1, M2)

        # TODO: check if this is correct
        # self.rotation is of R_C_W and we want R_C2_W=(R_C_W)^T @ R_C2_C1^T)^T
        # = R_C2_C1 @ R_C1_W
        cam2.rotation = R_CAM2_CAM1 @ self.rotation

        cam2.translation = c2T_CAM2_CAM1 + (R_CAM2_CAM1 @ self.translation)

        M = np.eye(4)
        M[:3, :3] = self.rotation.T
        M[:3, 3] = -self.rotation.T @ self.translation[:, 0]
        print(M)

        P = M @ P

        return P

    def draw_camera_wireframe(self, f, size, cam_name, color="black"):
        import plotly.graph_objects as go

        p1_c = np.array([-size / 2, -size / 2, f])
        p2_c = np.array([size / 2, -size / 2, f])
        p3_c = np.array([size / 2, size / 2, f])
        p4_c = np.array([-size / 2, size / 2, f])

        p1_w = self.rotation.T @ (p1_c - self.translation[:, 0])
        p2_w = self.rotation.T @ (p2_c - self.translation[:, 0])
        p3_w = self.rotation.T @ (p3_c - self.translation[:, 0])
        p4_w = self.rotation.T @ (p4_c - self.translation[:, 0])

        center_point = self.get_center_in_world()
        print(center_point.shape)

        # draw camera wireframe
        camera_wireframe = go.Scatter3d(
            x=[p1_w[0], p2_w[0], p3_w[0], p4_w[0], p1_w[0]],
            y=[p1_w[1], p2_w[1], p3_w[1], p4_w[1], p1_w[1]],
            z=[p1_w[2], p2_w[2], p3_w[2], p4_w[2], p1_w[2]],
            mode="lines",
            name=cam_name,
            line=dict(color=color, width=4),
            legendgroup=cam_name,
            showlegend=True,
        )

        center_line1 = go.Scatter3d(
            x=[center_point[0], p1_w[0]],
            y=[center_point[1], p1_w[1]],
            z=[center_point[2], p1_w[2]],
            mode="lines",
            name=cam_name + "line1",
            line=dict(color=color, width=4),
            legendgroup=cam_name,
            showlegend=False,
        )
        center_line2 = go.Scatter3d(
            x=[center_point[0], p2_w[0]],
            y=[center_point[1], p2_w[1]],
            z=[center_point[2], p2_w[2]],
            mode="lines",
            name=cam_name + "line2",
            line=dict(color=color, width=4),
            legendgroup=cam_name,
            showlegend=False,
        )
        center_line3 = go.Scatter3d(
            x=[center_point[0], p3_w[0]],
            y=[center_point[1], p3_w[1]],
            z=[center_point[2], p3_w[2]],
            mode="lines",
            name=cam_name + "line3",
            line=dict(color=color, width=4),
            legendgroup=cam_name,
            showlegend=False,
        )
        center_line4 = go.Scatter3d(
            x=[center_point[0], p4_w[0]],
            y=[center_point[1], p4_w[1]],
            z=[center_point[2], p4_w[2]],
            mode="lines",
            name=cam_name + "line4",
            line=dict(color=color, width=4),
            legendgroup=cam_name,
            showlegend=False,
        )

        lines = [
            camera_wireframe,
            center_line1,
            center_line2,
            center_line3,
            center_line4,
        ]

        return lines
