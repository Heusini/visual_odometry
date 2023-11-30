import numpy as np
from typing import NamedTuple

# equivalent to S^{i} from the project statement
class FrameState(NamedTuple):
    iteration : int
    keypoints : np.ndarray # keypoints in the camera frame
    landmarks : np.ndarray # keypoints in the world frame
    candidate_keypoints : np.ndarray


def process_frame(
    state_prev : FrameState,
    image_prev : np.ndarray,
    image_curr : np.ndarray,
) -> FrameState:
   
   pass 