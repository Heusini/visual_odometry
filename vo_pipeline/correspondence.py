from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from feature_detection import feature_detection, feature_matching
from helpers import load_images

def correspondence(
    features_cam1,
    features_cam2,
    feature_matching_threshold: float = 0.99,
) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    """
    Associated task:

        3.2. Establish keypoint correspondences between these two frames using either patch matching
        (exercise 3) or KLT (exercise 8, consider using the intermediate frames as well).

    Parameters
    ----------
    features_cam1 : Tuple of keypoints and descriptors of camera1
    features_cam2 : Tuple of keypoints and descriptors of camera2

    Returns
    -------
    (np.ndarray, np.ndarray)
        Two arrays of shape (K, 2) where K is the number of keypoints.
    """
    kp1, des1 = features_cam1
    kp2, des2 = features_cam2
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
    from camera import Camera
    imgs = load_images("data/kitti/05/image_0/", start=0, end=2)

    K = np.array(
        [
            [7.188560000000e02, 0, 6.071928000000e02],
            [0, 7.188560000000e02, 1.852157000000e02],
            [0, 0, 1],
        ]
    )

    img1 = imgs[0]
    img2 = imgs[-1]

    cam1 = Camera(1, K, img1)
    cam2 = Camera(2, K, img2)
    cam1.calculate_features()
    cam2.calculate_features()
    (keypoints_a, keypoints_b), (_, _) = correspondence(
        cam1.features, cam2.features, 0.99
    )

    # plot 32 keypoints in 4 x 8 grid
    s = 32
    fig, axs = plt.subplots(4, 8)
    for i in range(32):
        p_a_x = int(keypoints_a[i, 0])
        p_a_y = int(keypoints_a[i, 1])
        p_b_x = int(keypoints_b[i, 0])
        p_b_y = int(keypoints_b[i, 1])

        start_x = min(max(p_a_x - s, 0), img1.shape[0])
        end_x = min(max(p_a_x + s, 0), img1.shape[0])
        start_y = min(max(p_a_y - s, 0), img1.shape[1])
        end_y = min(max(p_a_y + s, 0), img1.shape[1])

        img1_patch = np.zeros((2 * s, 2 * s))
        img1_patch[: end_x - start_x, : end_y - start_y] = img1[
            start_x:end_x, start_y:end_y
        ]
       
        start_x = min(max(p_b_x - s, 0), img2.shape[0])
        end_x = min(max(p_b_x + s, 0), img2.shape[0])
        start_y = min(max(p_b_y - s, 0), img2.shape[1])
        end_y = min(max(p_b_y + s, 0), img2.shape[1])

        img2_patch = np.zeros((2 * s, 2 * s))
        img2_patch[: end_x - start_x, : end_y - start_y] = img2[
            start_x:end_x, start_y:end_y
        ]

        together = np.hstack((img1_patch, img2_patch))

        axs[i // 8, i % 8].imshow(together)

        # add red line between the two patches in matplotlib
        axs[i // 8, i % 8].plot(
            [2 * s, 2 * s], [0, 2 * s], color="red", linestyle="dashed", linewidth=1
        )
    plt.show()


if __name__ == "__main__":
    visualize_correspondence()
