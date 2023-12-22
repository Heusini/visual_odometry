import numpy as np
import cv2 as cv

def klt(
    pts_i : np.ndarray,
    img_i : np.ndarray,
    img_j : np.ndarray) -> (np.ndarray, np.ndarray):

    # convert images to greyscale if necessary
    if len(img_i.shape) == 3:
        img_i = cv.cvtColor(img_i, cv.COLOR_BGR2GRAY)
    if len(img_j.shape) == 3:
        img_j = cv.cvtColor(img_j, cv.COLOR_BGR2GRAY)

    pts_i = pts_i.astype(np.float32)
    pts_i = np.reshape(pts_i, (pts_i.shape[0], 1, 2))
    pts_j = np.zeros_like(pts_i)

    status = np.zeros((pts_j.shape[0], 1), dtype=np.uint8)

    cv.calcOpticalFlowPyrLK(
        img_i,
        img_j,
        pts_i,
        pts_j,
        status)
    
    mask = status.ravel() == 1

    return pts_j, mask
