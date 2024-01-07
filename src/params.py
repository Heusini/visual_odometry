from enum import Enum
import cv2 as cv
class FeatureDetector:
    KLT = 0
    SIFT = 1

DATASET = 0
class KITTIParams:
    class SIFT_PARAMS:
        MATCHING_THRESHOLD = 0.8
        NFEATURES = 0
        NOCTAVELAYERS = 4
        CONTRASTTHRESHOLD = 0.03
        EDGETHRESHOLD = 10
        SIGMA = 1.6

    class KLT_PARMS:
        WIN_SIZE = (21, 21)  # Typical default window size
        MAX_LEVEL = 8        # Default maximum pyramid level
        CRITERIA = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.001)  # Termination criteria
        MIN_EIG_THRESHOLD = 0.001  # Minimum eigenvalue threshold

    class INIT_PARAMS:
        class RANSAC_PARAMS_F:
            THRESHOLD = 0.1
            CONFIDENCE = 0.9999
            NUM_ITERATIONS = 100

        BASELINE_FRAME_INDICES = [0, 10]  # Frames used to init pointcloud
        MATCHER = FeatureDetector.SIFT
        MAX_DEPTH_DISTANCE = 50

    class CONT_PARAMS:
        class PNP_RANSAC_PARAMS:
            THRESHOLD = 1
            CONFIDENCE = 0.9999
            NUM_ITERATIONS = 100

        MAX_TRACK_LENGTH = 10
        MIN_TRACK_LENGTH = 6
        SAME_KEYPOINTS_THRESHOLD = 2
        MAX_DEPTH_DISTANCE = 50
        BASELINE_SIGMA = 0.07
        MIN_NUM_LANDMARKS = float('inf') # setting to inf deactivates baseline_sigma basically

class MALAGAParams:
    class SIFT_PARAMS:
        MATCHING_THRESHOLD = 0.6
        NFEATURES = 0
        NOCTAVELAYERS = 3
        CONTRASTTHRESHOLD = 0.04
        EDGETHRESHOLD = 10
        SIGMA = 1.6

    class KLT_PARMS:
        WIN_SIZE = (21, 21)  # Typical default window size
        MAX_LEVEL = 8        # Default maximum pyramid level
        CRITERIA = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.001)  # Termination criteria
        MIN_EIG_THRESHOLD = 0.01  # Minimum eigenvalue threshold

    class INIT_PARAMS:
        class RANSAC_PARAMS_F:
            THRESHOLD = 0.1
            CONFIDENCE = 0.9999
            NUM_ITERATIONS = 100

        BASELINE_FRAME_INDICES = [0, 4]  # Frames used to init pointcloud
        MATCHER = FeatureDetector.SIFT
        MAX_DEPTH_DISTANCE = 10000

    class CONT_PARAMS:
        class PNP_RANSAC_PARAMS:
            THRESHOLD = 1
            CONFIDENCE = 0.9999
            NUM_ITERATIONS = 10000

        MAX_TRACK_LENGTH = 10
        MIN_TRACK_LENGTH = 6
        SAME_KEYPOINTS_THRESHOLD = 2
        MAX_DEPTH_DISTANCE = 50
        BASELINE_SIGMA = 0.15
        MIN_NUM_LANDMARKS = 2000
    
class PARKINGParams:
    class SIFT_PARAMS:
        MATCHING_THRESHOLD = 0.8
        NFEATURES = 0
        NOCTAVELAYERS = 4
        CONTRASTTHRESHOLD = 0.03
        EDGETHRESHOLD = 10
        SIGMA = 1.6

    class KLT_PARMS:
        WIN_SIZE = (31, 31)  # Typical default window size
        MAX_LEVEL = 3        # Default maximum pyramid level
        CRITERIA = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 6)  # Termination criteria
        MIN_EIG_THRESHOLD = 0.01  # Minimum eigenvalue threshold

    class INIT_PARAMS:
        class RANSAC_PARAMS_F:
            THRESHOLD = 0.1
            CONFIDENCE = 0.9999
            NUM_ITERATIONS = 10000

        BASELINE_FRAME_INDICES = [0, 6]  # Frames used to init pointcloud
        MATCHER = FeatureDetector.SIFT
        MAX_DEPTH_DISTANCE = 50

    class CONT_PARAMS:
        class PNP_RANSAC_PARAMS:
            THRESHOLD = 1
            CONFIDENCE = 0.9999
            NUM_ITERATIONS = 10000

        MAX_TRACK_LENGTH = 10
        MIN_TRACK_LENGTH = 7
        SAME_KEYPOINTS_THRESHOLD = 2
        MAX_DEPTH_DISTANCE = 50
        BASELINE_SIGMA = 0.1
        MIN_NUM_LANDMARKS = float('inf')

def get_params():
    if DATASET == 0:
        return KITTIParams
    elif DATASET == 1:
        return MALAGAParams
    else:
        return PARKINGParams

def which_dataset(dataset: int):
    global DATASET
    DATASET = dataset