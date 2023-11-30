import numpy as np

def ransac(
    keypoints_a : np.ndarray, 
    keypoints_b : np.ndarray, 
    images : np.ndarray, 
    threshold : float = 0.1, 
    iterations : int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC filtering of keypoints.
    """
)