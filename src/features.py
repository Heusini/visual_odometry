import cv2 as cv
import numpy as np
from typing import NamedTuple, List, Any

from cv2 import calcOpticalFlowPyrLK

# this is identical to the openCV KeyPoint class -> do not change
class Keypoint(NamedTuple):
    pt : np.ndarray # 2D location of the keypoint
    size : float # size of the keypoint
    angle : float # orientation of the keypoint
    response : float # strength of the keypoint
    octave : int # octave of the keypoint
    class_id : int # id of the keypoint

class Features(NamedTuple):
    keypoints : List[Keypoint]
    descriptors : np.ndarray

    def get_positions(self) -> np.ndarray:
        ls = [ k.pt for k in self.keypoints ]
        return np.array(ls)
    

def detect_features(img) -> Features:
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return Features(kp, des)

def match_features(
    feature_set_i : Features,
    feature_set_j : Features,
    threshold=0.99) -> (Features, Features):

    bf = cv.BFMatcher()
    matches = bf.knnMatch(
        feature_set_i.descriptors, feature_set_j.descriptors, k=2)

    matches_filtered = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            matches_filtered.append(m)

    matching_kps_i = [feature_set_i.keypoints[m.queryIdx] for m in matches_filtered]
    matching_kps_j = [feature_set_j.keypoints[m.trainIdx] for m in matches_filtered]

    matching_des_i = [feature_set_i.descriptors[m.queryIdx] for m in matches_filtered]
    matching_des_j = [feature_set_j.descriptors[m.trainIdx] for m in matches_filtered]

    matching_des_i = np.array(matching_des_i)
    matching_des_j = np.array(matching_des_j)

    matching_features_i = Features(matching_kps_i, matching_des_i)
    matching_features_j = Features(matching_kps_j, matching_des_j)

    return matching_features_i, matching_features_j

if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    from helpers import load_images
    from matplotlib.patches import ConnectionPatch

    show_lines = False
    show_sift_features = True

    # <<<<<<< HEAD
    #     img1 = cv.imread("data/kitti/05/image_0/000000.png")
    #     img2 = cv.imread("data/kitti/05/image_0/000003.png")
    #
    # =======
    imgs = load_images("data/parking/images/", start=1, end=3)
    img_i = imgs[0]
    img_j = imgs[-1]
    t1 = time.time()
    features_i = detect_features(img_i)
    t2 = time.time()
    features_j = detect_features(img_j)
    t3 = time.time()
    
    matching_features_i, matching_features_j = match_features(features_i, features_j, threshold=0.99)
    print(f"Number of matching features: {len(matching_features_i.keypoints)}")

    t4 = time.time()
    print(f"Calculating features img1 took: {t2-t1}")
    print(f"Calculating features img2 took: {t3-t2}")
    print(f"Calculating matching img1-img2 took: {t4-t3}")

    fig, ax = plt.subplots(1, 2)
    if show_sift_features:
        ax[0].imshow(cv.drawKeypoints(img_i, matching_features_j.keypoints, None))
        ax[1].imshow(cv.drawKeypoints(img_j, matching_features_j.keypoints, None))
    else:
        ax[0].imshow(img_i)
        ax[1].imshow(img_j)

    positions_i = matching_features_i.get_positions()
    positions_j = matching_features_j.get_positions()

    print(positions_i)

    ax[0].scatter(positions_i[:, 0], positions_i[:, 1], c="r", s=1)
    ax[1].scatter(positions_j[:, 0], positions_j[:, 1], c="r", s=1)

    if show_lines:
        for i in range(len(matching_features_i)):
            conn = ConnectionPatch(
                xyA=(positions_i[i, 0], positions_i[i, 1]),
                xyB=(positions_j[i, 0], positions_j[i, 1]),
                coordsA="data",
                coordsB="data",
                axesA=ax[0],
                axesB=ax[1],
                color="green",
            )
            ax[1].add_artist(conn)
            conn.set_in_layout(False)

    plt.show()
