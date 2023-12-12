import numpy as np
from typing import NamedTuple, List
from feature_detection import feature_detection, feature_matching
from sfm import sfm, Camera

# equivalent to S^{i} from the project statement
class FrameState(NamedTuple):
    iteration : int
    keypoints : np.ndarray # keypoints positions
    descriptors : np.ndarray # descriptors in the camera frame
    landmarks : np.ndarray # keypoints in the world frame
    camera : Camera # camera object
    # TODO: extend as described in 4.3

def process_frame(
    state_prev : FrameState,
    image_curr : np.ndarray,
    K : np.ndarray,
) -> FrameState:
   
    kp_curr, des_curr = feature_detection(image_curr)

    matches = feature_matching(state_prev.descriptors, des_curr, threshold=0.99)

    xs = [kp_curr[m.trainIdx].pt[0] for m in matches]
    ys = [kp_curr[m.trainIdx].pt[1] for m in matches]

    kp_curr = np.stack((np.asarray(xs), np.asarray(ys)), axis=-1)

    n_keypoints_prev = state_prev.keypoints.shape[0]

    kp_curr = kp_curr[:n_keypoints_prev]

    P, cam_prev, cam_curr = sfm(kp_curr, state_prev.keypoints, K)
   
    return FrameState(
        state_prev.iteration + 1,
        kp_curr,
        des_curr,
        P,
        cam_curr,
    )