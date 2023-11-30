import numpy as np

def sfm(
    keypoints_a: np.ndarray,
    keypoints_b: np.ndarray,
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> (np.ndarray, (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    """
    Associated task:
        3.3. Estimate the relative pose between the frames and triangulate a point cloud of 3D landmarks
        (exercise 6)

    Parameters
    ----------
    keypoints_a : np.ndarray
        Shape is (K, 2) where K is the number of keypoints.
    keypoints_b : np.ndarray
        Shape is (K, 2) where K is the number of keypoints.

    Returns ( P , (R, T), (K_a, K_b) )
    -------
    P : np.ndarray
        3D Keypoints

    R : np.ndarray
        Rotation matrix

    T : np.ndarray
        Translation matrix

    K_a : np.ndarray
        Camera intrinsics for first image

    K_b : np.ndarray
        Camera intrinsics for second image
    """
    
    pass