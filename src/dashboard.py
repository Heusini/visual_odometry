import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def create_default_dashboard():
    fig = plt.figure(figsize=(10, 10), layout="constrained")
    gs0 = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs0[0, :])
    ax2 = fig.add_subplot(gs0[1, 0])
    ax3 = fig.add_subplot(gs0[1, 1])

    return fig, ax1, ax2, ax3


def cv_scatter(img, points, color, radius, thicknes) -> np.ndarray:
    for i in range(points.shape[1]):
        cv.circle(
            img,
            (int(points[0, i]), int(points[1, i])),
            radius=radius,
            color=color,
            thickness=thicknes,
        )
    return img


if __name__ == "__main__":
    test_image_path = "data/kitti/05/image_0/000000.png"
    img = cv.imread(test_image_path, cv.COLOR_BGR2RGB)

    fig, ax1, _, _ = create_default_dashboard()
    ax1.imshow(img)

    plt.show()
