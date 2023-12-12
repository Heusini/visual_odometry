import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from feature_detection import feature_detection, feature_matching
from helpers import load_images


def correspondence(
    imgs: np.ndarray,
    feature_matching_threshold : float = 0.99,
) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    
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
    (np.ndarray, np.ndarray)
        Two arrays of shape (K, 2) where K is the number of keypoints.
    """
    kp1, des1 = feature_detection(imgs[0])
    kp2, des2 = feature_detection(imgs[-1])

    matches = feature_matching(des1, des2, threshold=feature_matching_threshold)
    x_values1 = [kp1[m.queryIdx].pt[0] for m in matches]
    y_values1 = [kp1[m.queryIdx].pt[1] for m in matches]
    x_values2 = [kp2[m.trainIdx].pt[0] for m in matches]
    y_values2 = [kp2[m.trainIdx].pt[1] for m in matches]

    p1_x = np.asarray(x_values1)
    p1_y = np.asarray(y_values1)
    p2_x = np.asarray(x_values2)
    p2_y = np.asarray(y_values2)

    p1 = np.stack((p1_x, p1_y), axis=-1)
    p2 = np.stack((p2_x, p2_y), axis=-1)
    return (p1, p2), (des1, des2)


def visualize_correspondence():
    imgs = load_images("data/kitti/05/image_0/", start=0, end=2)

    (keypoints_a, keypoints_b), (des_a, des_b) = correspondence(imgs)

    # plot 8 keypoints with a window size of 5 pixels
    s = 15
    fig, axs = plt.subplots(2, 8)
    for i in range(8):
        p_a_x = int(keypoints_a[i, 0])
        p_a_y = int(keypoints_a[i, 1])
        p_b_x = int(keypoints_b[i, 0])
        p_b_y = int(keypoints_b[i, 1])
        axs[0, i].imshow(
            imgs[0, p_a_x - s : p_a_x + s, p_a_y - s : p_a_y + s], cmap="gray"
        )
        axs[1, i].imshow(
            imgs[-1, p_b_x - s : p_b_x + s, p_b_y - s : p_b_y + s], cmap="gray"
        )

    plt.show()


if __name__ == "__main__":
    visualize_correspondence()
