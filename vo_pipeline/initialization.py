import numpy as np
import vo_pipeline as vo

def initialization(images : np.ndarray, I_1 : int, K : np.ndarray) -> vo.FrameState:
    
    keypoints_a, keypoints_b = vo.correspondence(images)

    # TODO: add ransac filering of keypoints here

    landmarks, camera_a, camera_b = vo.sfm(keypoints_a, keypoints_b, K)

    return vo.FrameState(I_1, keypoints_b, landmarks)