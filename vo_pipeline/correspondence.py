import numpy as np

def correspondence(imgs : np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Associated task: 

        3.2. Establish keypoint correspondences between these two frames using either patch matching
        (exercise 3) or KLT (exercise 8, consider using the intermediate frames as well).

    Parameters
    ----------
    imgs : np.ndarray
        Shape is (N, H, W) where N is the number of images and
        H and W are the height and width of the images.

    Returns
    -------
    np.ndarray
        Shape is (K, 2, 2) because for each keypoint we its uv coordinates
        in the first and last image.
    """

    return np.random.rand(100, 2), np.random.rand(100, 2)
    