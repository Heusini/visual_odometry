import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class Dashboard:
    def __init__(self):
        self.fig, self.fig_axes = create_default_dashboard()

    def update_axis(self, ax_num, points, color='b'):
        self.fig_axes[ax_num].clear()
        self.fig_axes[ax_num].scatter(points[0], points[1], s=20, c=color)
        x_min = np.min(points[0])
        x_max = np.max(points[0])
        y_min = np.min(points[1])
        y_max = np.max(points[1])
        self.fig_axes[ax_num].set_xlim(x_min - 0.5, x_max + 0.5)
        self.fig_axes[ax_num].set_ylim(y_min - 0.5, y_max + 0.5)

    def update_image(self, ax_num, img, points_2d):
        color = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for i in range(len(points_2d)):
            img = cv_scatter(img, points_2d[i], color[i], 3, -1)
        self.fig_axes[ax_num].imshow(img)

    def update_cams(self, ax_num, cam_points):
        self.fig_axes[ax_num].clear()
        x_cams = []
        y_cams = []
        for cam in cam_points[:-1]:
            centerpoint = cam @ np.array([0, 0, 0, 1])
            self.fig_axes[ax_num].scatter(centerpoint[0], centerpoint[2], s=20, c="r")
            x_cams.append(centerpoint[0])
            y_cams.append(centerpoint[2])
        centerpoint = cam_points[-1] @ np.array([0, 0, 0, 1])
        self.fig_axes[ax_num].scatter(centerpoint[0], centerpoint[2], s=20, c="g")
        x_cams.append(centerpoint[0])
        y_cams.append(centerpoint[2])
        x_min = np.min(x_cams)
        x_max = np.max(x_cams)
        y_min = np.min(y_cams)
        y_max = np.max(y_cams)
        self.fig_axes[ax_num].set_xlim(x_min - 0.5, x_max + 0.5)
        self.fig_axes[ax_num].set_ylim(y_min - 0.5, y_max + 0.5)

def create_default_dashboard():
    fig = plt.figure(figsize=(10, 10), layout="constrained")
    gs0 = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs0[0, :])
    ax2 = fig.add_subplot(gs0[1, 0])
    ax3 = fig.add_subplot(gs0[1, 1])

    fig_axes= [ax1, ax2, ax3]

    return fig, fig_axes


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

