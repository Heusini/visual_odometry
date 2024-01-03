import numpy as np
import cv2 as cv

def klt(
    pts_i : np.ndarray, # [N, 2] array of 2D keypoints
    img_i : np.ndarray,
    img_j : np.ndarray,
    lk_params : dict = dict(winSize=(15, 15), maxLevel=10, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))) -> (np.ndarray, np.ndarray):

    # convert images to greyscale if necessary
    if len(img_i.shape) == 3:
        img_i = cv.cvtColor(img_i, cv.COLOR_BGR2GRAY)
    if len(img_j.shape) == 3:
        img_j = cv.cvtColor(img_j, cv.COLOR_BGR2GRAY)

    pts_i = pts_i.astype(np.float32)
    pts_i = np.reshape(pts_i, (pts_i.shape[0], 1, 2))
    pts_j = np.zeros_like(pts_i)

    status = np.zeros((pts_j.shape[0], 1), dtype=np.uint8)

    # Parameters for Lucas-Kanade optical flow
    # maxlevel defines how many times we down scale the original image (at max), especially important to tune
    # when skipping multiple frames (large frame to frame motion)
    # Also make the winsize smaller if you increase the maxlevel
    # citerias are standard, works also without it. Just added it to show that there
    # are other parameters that we could consider to optimize KLT.

    pts_j, status, error = cv.calcOpticalFlowPyrLK(
        img_i,
        img_j,
        pts_i,
        None,
        **lk_params)

    
    mask = status.ravel() == 1

    pts_j = np.reshape(pts_j, (pts_j.shape[0], 2))

    return pts_j, mask
