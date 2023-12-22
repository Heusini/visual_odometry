import cv2 as cv
import numpy as np
from camera import Camera
from camera_pose import plot_plotly
from helpers import load_images
from enum import Enum
from state import FrameSate
import utils.geometry as geom

from correspondence import correspondence
from utils.decompose_essential_matrix import decomposeEssentialMatrix
from utils.disambiguate_relative_pose import disambiguateRelativePose
from utils.linear_triangulation import linearTriangulation

def pnp():
    pass

class DataSetEnum(Enum):
    KITTI = "kitti"
    PARKING = "parking"
    MALAGA = "malaga"

# def pnp(
#     state_i : FrameSate,
#     state_j : FrameSate): 



#     F, mask_f = geom.get_fundamental_matrix(matched_points_cam1, matched_points_cam2)
#     E = geom.essential_matrix_from_fundamental_matrix(F, self.K)
#     p1 = np.hstack(
#         [matched_points_cam1, np.ones((matched_points_cam1.shape[0], 1))]
#     ).T
#     p2 = np.hstack(
#         [matched_points_cam2, np.ones((matched_points_cam2.shape[0], 1))]
#     ).T
#     p1 = p1[:, mask_f.ravel() == 1]
#     p2 = p2[:, mask_f.ravel() == 1]
#     R, T = decomposeEssentialMatrix(E)
#     R_cam2_to_cam1, T_cam2_to_cam1 = disambiguateRelativePose(
#         R, T, p1, p2, self.K, self.K
#     )

#     M1 = self.K @ np.eye(3, 4)
#     M2 = cam2.K @ np.c_[R_cam2_to_cam1, T_cam2_to_cam1]
#     P_cam1 = linearTriangulation(p1, p2, M1, M2)

#     # filer points behind camera and far away
#     max_distance = 100
#     mask = np.logical_and(P_cam1[2, :] > 0, np.abs(np.linalg.norm(P_cam1, axis=0)) < max_distance)
#     P_cam1 = P_cam1[:, mask]

#     cam2.R_to_cam = np.linalg.inv(R_cam2_to_cam1) @ self.R_to_cam
#     cam2.T_to_cam = self.T_to_cam + np.reshape(T_cam2_to_cam1, (3, 1))
#     print(f"cam2.T_to_cam: {cam2.T_to_cam}")
#     print(f"cam2.R_to_cam: {cam2.R_to_cam}")
#     M_to_cam = self.M_to_cam()
#     M_to_world = np.linalg.inv(M_to_cam)

#     P_world = M_to_world @ P_cam1

#     return P_world

    

if __name__ == "__main__":
    
    dataset = DataSetEnum.KITTI
    stride = 4
    steps = 5
    
    K_kitty = np.array(
        [
            [7.188560000000e02, 0, 6.071928000000e02],
            [0, 7.188560000000e02, 1.852157000000e02],
            [0, 0, 1],
        ]
    )
    K_parking = np.array([[331.37, 0, 320], [0, 369.568, 240], [0, 0, 1]])

    if dataset == DataSetEnum.KITTI:
        K = K_kitty
        imgs = load_images("data/kitti/05/image_0/", start=0, end=stride*steps)
    elif dataset == DataSetEnum.PARKING:
        K = K_parking
        imgs = load_images("data/parking/images/", start=0, end=stride*steps)

    cams = []

    for step in range(steps):
        img = imgs[int(step*stride)]
        cam = Camera(step, K, img)
        cam.calculate_features()
        cams.append(cam)

    Ps = []
    for step in range(steps - 1):
        cam_i = cams[step]
        cam_j = cams[step+1]
        (m1, m2), (_, _) = cam_i.calculate_matches(cam_j)
        P = cam_i.process_next_frame(cam_j, m1, m2)
        Ps.append(P)

    plot_plotly(Ps, cams)