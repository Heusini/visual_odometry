import numpy as np

def projectPoints(points_3d, K, D=np.zeros([4, 1])):
    """
    Projects 3d points to the image plane (3xN), given the camera matrix (3x3) and
    distortion coefficients (4x1).
    """
    # get image coordinates
    projected_points = np.matmul(K, points_3d[:, :, None]).squeeze(-1)
    projected_points /= projected_points[:, 2, None]

    return projected_points[:, :2]
