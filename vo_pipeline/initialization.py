import numpy as np
import vo_pipeline as vo

def initialization(images : np.ndarray, I_1 : int) -> vo.FrameState:
    
    keypoints_a, keypoints_b = vo.correspondence(images)

    # TODO: add ransac filering of keypoints here

    landmarks, (R, T), (K_a, K_b) = vo.sfm(keypoints_a, keypoints_b, images[0], images[-1])

    return vo.FrameState(I_1, keypoints_b, landmarks)
