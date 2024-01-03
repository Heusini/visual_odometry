import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from utils.utils import HomogMatrix2twist, twist2HomogMatrix
import matplotlib.pyplot as plt

def refine_camera_pose(keypoints: np.ndarray, landmarks: np.ndarray, C_guess, K, plot_debug: bool = False):
    P = keypoints.T
    X = landmarks.T

    R_C_W_guess = C_guess[:3, :3]
    t_C_W_guess = C_guess[:3, -1].reshape(-1, 1)
    # Initial guess for optimization solution
    x_init = HomogMatrix2twist(np.vstack([np.hstack([R_C_W_guess, t_C_W_guess]), [0, 0, 0, 1]]))

    # Define objective function
    def error_terms(x):
        # Ensure the input sizes are correct
        assert X.shape[1] == P.shape[1]

        # Obtain world to camera transformation matrix
        T_C_W = twist2HomogMatrix(x)

        # Perform projection of 3D landmarks to camera frame
        M_C_W = np.dot(K, T_C_W[:3, :])
        p_projected_hom = np.dot(M_C_W, np.vstack([X, np.ones((1, X.shape[1]))]))
        p_projected = p_projected_hom[:2, :] / p_projected_hom[2, :]

        # Compute reprojection error
        error = P - p_projected

        if plot_debug:
            plt.clf()
            plt.close()
            plt.plot(p_projected[0, :], p_projected[1, :], 'o')
            plt.plot(P[0, :], P[1, :], 'x')
            plt.axis('equal')
            plt.show()
            plt.waitforbuttonpress()
        
        return error.flatten()

    # Perform optimization
    options = {'max_nfev': 20, 'verbose': 2}
    result = least_squares(error_terms, x_init, **options)

    # Convert pose from twist to homogeneous matrices
    T_C_W_optim = twist2HomogMatrix(result.x)
    R_C_W_optim = T_C_W_optim[:3, :3]
    t_C_W_optim = T_C_W_optim[:3, -1].reshape(-1, 1)
    T_W_C_optim = np.hstack([
        R_C_W_optim.T,
        -R_C_W_optim.T @ t_C_W_optim
    ])
    T_W_C_optim = np.vstack([
        T_W_C_optim,
        [0, 0, 0, 1]
    ])
    return T_W_C_optim