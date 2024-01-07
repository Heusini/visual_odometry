from typing import List
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class Dashboard:
    def __init__(self):
        self.fig, self.fig_axes = create_default_dashboard()

    def set_limits(self, ax_num, x_lim, y_lim):
        x_min = x_lim[0]
        x_max = x_lim[1]
        y_min = y_lim[0]
        y_max = y_lim[1]
        self.fig_axes[ax_num].set_xlim(x_min - 0.5, x_max + 0.5)
        self.fig_axes[ax_num].set_ylim(y_min - 0.5, y_max + 0.5)

    def calculate_limits_from_points(self, points):
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for i in range(len(points)):
            x_min_new = np.min(points[i][0])
            x_max_new = np.max(points[i][0])
            y_min_new = np.min(points[i][1])
            y_max_new = np.max(points[i][1])
            if x_min_new < x_min:
                x_min = x_min_new
            if x_max_new > x_max:
                x_max = x_max_new
            if y_min_new < y_min:
                y_min = y_min_new
            if y_max_new > y_max:
                y_max = y_max_new

        return [x_min, x_max], [y_min, y_max]


    def update_axis(self, ax_num, points, color='b', size=20):
        self.fig_axes[ax_num].scatter(points[0], points[1], s=size, c=color)

    def update_axis_with_clear(self, ax_num, points, color='b', size=20):
        self.fig_axes[ax_num].clear()
        self.update_axis(ax_num, points, color, size)

    def update_image(self, ax_num, img, points_2d, colors: List[List[int]]):
        colors = colors
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for i in range(len(points_2d)):
            img = cv_scatter(img, points_2d[i], colors[i], 3, -1)
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

    def clear_axis(self, ax_num):
        self.fig_axes[ax_num].clear()

def create_default_dashboard():
    fig = plt.figure(figsize=(15, 10), layout="constrained")
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    fig_axes= [ax1, ax2, ax3, ax4]

    return fig, fig_axes


def cv_scatter(img, points, color, radius, thicknes) -> np.ndarray:
    for i in range(points.shape[0]):
        cv.circle(
            img,
            (int(points[i, 0]), int(points[i, 1])),
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

