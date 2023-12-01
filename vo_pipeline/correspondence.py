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
    (np.ndarray, np.ndarray) 
        Two arrays of shape (K, 2) where K is the number of keypoints.
    """

    return np.random.rand(10, 2), np.random.rand(10, 2)

    import correspondence as co

import matplotlib.pyplot as plt
import numpy as np
from helpers import load_images

def visualize_correspondence():
    imgs = load_images('data/kitti/05/image_0/', start=0, end=2)

    keypoints_a, keypoints_b = correspondence(imgs)

    # plot 8 keypoints with a window size of 5 pixels
    s = 5
    fig, axs = plt.subplots(2, 8)
    for i in range(8):
        p_a_x = int(keypoints_a[i, 0]) * imgs.shape[1]
        p_a_y = int(keypoints_a[i, 1]) * imgs.shape[2]
        p_b_x = int(keypoints_b[i, 0]) * imgs.shape[1]
        p_b_y = int(keypoints_b[i, 1]) * imgs.shape[2]
        axs[0, i].imshow(imgs[0, p_a_x-s:p_a_x+s, p_a_y-s:p_a_y+s], cmap='gray')
        axs[1, i].imshow(imgs[-1, p_b_x-s:p_b_x+s, p_b_y-s:p_b_y+s], cmap='gray')

    plt.show()

    # stack images horizontally
    both_imgs = np.hstack((imgs[0], imgs[1]))

    # plot all keypoints
    plt.imshow(both_imgs, cmap='gray')
    plt.scatter(keypoints_a[:, 1]  * imgs.shape[2], keypoints_a[:, 0] * imgs.shape[1], c='r')
    plt.scatter(keypoints_b[:, 1] * imgs.shape[2] + imgs.shape[2], keypoints_b[:, 0] * imgs.shape[1], c='r')

    # add lines between keypoints
    for i in range(keypoints_a.shape[0]):
        plt.plot([keypoints_a[i, 1] * imgs.shape[2], keypoints_b[i, 1] * imgs.shape[2] + imgs.shape[2]], [keypoints_a[i, 0] * imgs.shape[1], keypoints_b[i, 0] * imgs.shape[1]], c='r')

    plt.show()

visualize_correspondence()