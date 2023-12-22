import cv2 as cv
import numpy as np
from correspondence import correspondence
from utils.decompose_essential_matrix import decomposeEssentialMatrix
from utils.disambiguate_relative_pose import disambiguateRelativePose
from utils.linear_triangulation import linearTriangulation
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

# We use SRT transform order: Scale -> Rotation -> Translation    
class Camera:
    def __init__(
        self,
        id: int,
        K: np.ndarray,
        image,
        R_to_cam=np.eye(3),
        T_to_cam=np.zeros((3, 1)),
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
        # Rotation from world to camera
        self.R_to_cam = R_to_cam
        # Translation from world to camera. -T_to_cam is the camera origin in world coordinates
        self.T_to_cam = T_to_cam

        self.features = None
        self.matches = None
        self.point_cloud = None
        self.image = image
        self.K = K

    def R_to_world(self):
        return np.linalg.inv(self.R_to_cam)
    
    def T_to_world(self):
        return -self.T_to_cam
    
    def M_to_cam(self):
        M = np.eye(4)
        M[:3, :3] = self.R_to_cam
        M[:3, 3:] = self.T_to_cam
        return M
    
    def M_to_world(self):
        M = self.M_to_cam()
        return np.linalg.inv(M)
    
    def calculate_features(self):
        self.features = feature_detection(self.image)

    def calculate_matches(self, other: "Camera", threshold=0.7):
        return correspondence(self.features, other.features, threshold)

    def process_next_frame(
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
        R, T = decomposeEssentialMatrix(E)
        R_cam2_to_cam1, T_cam2_to_cam1 = disambiguateRelativePose(
            R, T, p1, p2, self.K, self.K
        )

        M1 = self.K @ np.eye(3, 4)
        M2 = cam2.K @ np.c_[R_cam2_to_cam1, T_cam2_to_cam1]
        P_cam1 = linearTriangulation(p1, p2, M1, M2)

        # filer points behind camera and far away
        max_distance = 100
        mask = np.logical_and(P_cam1[2, :] > 0, np.abs(np.linalg.norm(P_cam1, axis=0)) < max_distance)
        P_cam1 = P_cam1[:, mask]

        cam2.R_to_cam = np.linalg.inv(R_cam2_to_cam1) @ self.R_to_cam
        cam2.T_to_cam = self.T_to_cam + np.reshape(T_cam2_to_cam1, (3, 1))
        print(f"cam2.T_to_cam: {cam2.T_to_cam}")
        print(f"cam2.R_to_cam: {cam2.R_to_cam}")
        M_to_cam = self.M_to_cam()
        M_to_world = np.linalg.inv(M_to_cam)

        P_world = M_to_world @ P_cam1

        return P_world

    def draw_camera_wireframe(self, f, size, cam_name, color="black"):
        import plotly.graph_objects as go

        p1_c = np.array([-size / 2, -size / 2, f, 1.0])
        p2_c = np.array([size / 2, -size / 2, f, 1.0])
        p3_c = np.array([size / 2, size / 2, f, 1.0])
        p4_c = np.array([-size / 2, size / 2, f, 1.0])

        M_to_world = self.M_to_world()

        print(f"M_to_world: {M_to_world}")

        p1_w = M_to_world @ p1_c
        p2_w = M_to_world @ p2_c
        p3_w = M_to_world @ p3_c
        p4_w = M_to_world @ p4_c

        center_point = self.T_to_cam
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
