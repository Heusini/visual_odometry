import numpy as np
import scipy
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from utils.utils import twist2HomogMatrix
from utils.project_points import projectPoints


def runBA(hidden_state, observations, K):
    """
    Update the hidden state, encoded as explained in the problem statement, with 20 bundle adjustment iterations.
    """
    with_pattern = True
    hidden_state = hidden_state.astype(np.float32)
    observations = observations.astype(np.float32)
    K = K.astype(np.float32)

    pattern = None
    if with_pattern:
        num_frames = int(observations[0])
        num_observations = (observations.shape[0] - 2 - num_frames) / 3

        # Factor 2, one error for each x and y direction.
        num_error_terms = int(2 * num_observations)
        # Each error term will depend on one pose (6 entries) and one landmark position (3 entries),
        # so 9 nonzero entries per error term:
        pattern = scipy.sparse.lil_matrix((num_error_terms, hidden_state.shape[0]), dtype=np.int8)
        
        # Fill pattern for each frame individually:
        observation_i = 2  # iterator into serialized observations
        error_i = 0  # iterating frames, need another iterator for the error

        for frame_i in range(num_frames):
            num_keypoints_in_frame = int(observations[observation_i])
            # All errors of a frame are affected by its pose.
            pattern[error_i:error_i + 2 * num_keypoints_in_frame, frame_i*6:(frame_i + 1)*6] = 1

            # Each error is then also affected by the corresponding landmark.
            landmark_indices = observations[observation_i + 2 * num_keypoints_in_frame + 1:
                                            observation_i + 3 * num_keypoints_in_frame + 1]

            for kp_i in range(landmark_indices.shape[0]):
                pattern[error_i + kp_i * 2:error_i + (kp_i+1) * 2,
                        num_frames * 6 + int(landmark_indices[kp_i] - 1) * 3:num_frames * 6 + int(landmark_indices[kp_i]) * 3] = 1


            observation_i = observation_i + 1 + 3 * num_keypoints_in_frame
            error_i = error_i + 2 * num_keypoints_in_frame

        pattern = scipy.sparse.csr_matrix(pattern)

    def baError(hidden_state):
        plot_debug = False
        num_frames = int(observations[0])
        T_W_C = hidden_state[:num_frames * 6].reshape([-1, 6]).T
        p_W_landmarks = hidden_state[num_frames * 6:].reshape([-1, 3]).T

        error_terms = []
        
        # Iterator into the observations that are encoded as explained in the problem statement.
        observation_i = 1

        for i in range(num_frames):
            single_T_W_C = twist2HomogMatrix(T_W_C[:, i])
            num_frame_observations = int(observations[observation_i + 1])
            keypoints = np.flipud(observations[observation_i + 2:observation_i + 2 + num_frame_observations*2].reshape([-1, 2]).T)

            landmark_indices = observations[observation_i + 2 + num_frame_observations*2:observation_i + 2 + num_frame_observations * 3]
            
            # Landmarks observed in this specific frame.
            p_W_L = p_W_landmarks[:, landmark_indices.astype(np.int) - 1]
            
            # Transforming the observed landmarks into the camera frame for projection.
            T_C_W = np.linalg.inv(single_T_W_C)
            p_C_L = np.matmul(T_C_W[:3, :3], p_W_L.transpose(1, 0)[:, :, None]).squeeze(-1) + T_C_W[:3, -1]

            # From exercise 1.
            projections = projectPoints(p_C_L, K)
            
            # Can be used to verify that the projections are reasonable.
            if plot_debug:
                plt.clf()
                plt.close()
                plt.plot(projections[:, 0], projections[:, 1], 'o')
                plt.plot(keypoints[0, :], keypoints[1, :], 'x')
                plt.axis('equal')
                plt.show()
                plt.waitforbuttonpress()

            error_terms.append(keypoints.transpose(1, 0) - projections)
            observation_i = observation_i + num_frame_observations * 3 + 1

        return np.concatenate(error_terms).flatten()

    res_1 = least_squares(baError, hidden_state, max_nfev=20, verbose=2, jac_sparsity=pattern)
    hidden_state = res_1.x

    return hidden_state
