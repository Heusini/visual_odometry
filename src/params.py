from enum import Enum
from features import FeatureDetector
import cv2 as cv

class KITTIParams(Enum):
    class SIFT_PARAMS(Enum):
        MATCHING_THRESHOLD = 0.8
        NFEATURES = 0
        NOCTAVELAYERS = 3
        CONTRASTTHRESHOLD = 0.04
        EDGETHRESHOLD = 10
        SIGMA = 1.6

    class KLT_PARMS(Enum):
        WIN_SIZE = (21, 21)  # Typical default window size
        MAX_LEVEL = 8        # Default maximum pyramid level
        CRITERIA = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.001)  # Termination criteria
        MIN_EIG_THRESHOLD = 0.001  # Minimum eigenvalue threshold

    class INIT_PARAMS(Enum):
        class RANSAC_PARAMS_F(Enum):
            THRESHOLD = 0.1
            CONFIDENCE = 0.9999
            NUM_ITERATIONS = 100

        BASELINE_FRAME_INDICES = [0, 2]  # Frames used to init pointcloud
        MATCHER = FeatureDetector.KLT
        MAX_DEPTH_DISTANCE = 100

    class CONT_PARAMS(Enum):
        class PNP_RANSAC_PARAMS(Enum):
            THRESHOLD = 1
            CONFIDENCE = 0.9999
            NUM_ITERATIONS = 100

        ANGLE_THRESHOLD = 0.0436
        MAX_TRACK_LENGTH = 10
        MIN_TRACK_LENGTH = 6
        SAME_KEYPOINTS_THRESHOLD = 2
        MAX_DEPTH_DISTANCE = 50

