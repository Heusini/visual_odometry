import cv2 as cv
import numpy as np


def feature_detection(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    return kp, des


def feature_matching(des1, des2, threshold=0.75):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)

    return good


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    from helpers import load_images

    show_lines = True
    show_sift_features = False

    imgs = load_images("data/kitti/05/image_0/", start=1, end=3)
    img1 = imgs[0]
    img2 = imgs[-1]
    t1 = time.time()
    kp1, des1 = feature_detection(img1)
    t2 = time.time()
    kp2, des2 = feature_detection(img2)
    t3 = time.time()

    good = feature_matching(des1, des2, 0.6)
    t4 = time.time()
    print(f"Calculating features img1 took: {t2-t1}")
    print(f"Calculating features img2 took: {t3-t2}")
    print(f"Calculating matching img1-img2 took: {t4-t3}")

    fig, ax = plt.subplots(1, 2)
    if show_sift_features:
        ax[0].imshow(cv.drawKeypoints(img1, kp1, None))
        ax[1].imshow(cv.drawKeypoints(img2, kp2, None))
    else:
        ax[0].imshow(img1)
        ax[1].imshow(img2)
    x_values1 = [kp1[g.queryIdx].pt[0] for g in good]
    y_values1 = [kp1[g.queryIdx].pt[1] for g in good]
    x_values2 = [kp2[g.trainIdx].pt[0] for g in good]
    y_values2 = [kp2[g.trainIdx].pt[1] for g in good]

    ax[0].scatter(x_values1, y_values1, c="r", s=1)
    ax[1].scatter(x_values2, y_values2, c="r", s=1)

    if show_lines:
        for i in range(len(good)):
            conn = ConnectionPatch(
                xyA=(x_values1[i], y_values1[i]),
                xyB=(x_values2[i], y_values2[i]),
                coordsA="data",
                coordsB="data",
                axesA=ax[0],
                axesB=ax[1],
                color="green",
            )
            ax[1].add_artist(conn)
            conn.set_in_layout(False)

    plt.show()
