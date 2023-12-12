import numpy as np
from feature_detection import feature_detection, feature_matching
from sfm import sfm

def process_frame(
    state_prev : dict,
    image_curr : np.ndarray,
) -> dict:
    
    keypoints_prev = np.array(state_prev['keypoints'])
    descriptors_prev = np.array(state_prev['descriptors'], dtype=np.float32)
    camera_rotation_prev = np.array(state_prev['camera_rotation'])
    camera_translation_prev = np.array(state_prev['camera_translation'])
    K = np.array(state_prev['K'])

    kp_curr, des_curr = feature_detection(image_curr)

    matches = feature_matching(descriptors_prev, des_curr, threshold=0.99)

    xs = [kp_curr[m.trainIdx].pt[0] for m in matches]
    ys = [kp_curr[m.trainIdx].pt[1] for m in matches]

    kp_curr = np.stack((np.asarray(xs), np.asarray(ys)), axis=-1)

    n_keypoints_prev = keypoints_prev.shape[0]

    kp_curr = kp_curr[:n_keypoints_prev]

    P, cam_prev, cam_curr = sfm(kp_curr, keypoints_prev, K)

    return {
        'K' : K,
        'img' : image_curr,
        'iteration' : state_prev['iteration'] + 1,
        'keypoints' : kp_curr,
        'keypoints_prev' : keypoints_prev,
        'descriptors' : des_curr,
        'landmarks' : P,
        'camera_rotation' : cam_curr.rotation,
        'camera_translation' : cam_curr.translation,
    }