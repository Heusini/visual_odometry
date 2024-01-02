import numpy as np
from scipy.linalg import expm, logm


def cross2Matrix(x):
    """ Antisymmetric matrix corresponding to a 3-vector
     Computes the antisymmetric matrix M corresponding to a 3-vector x such
     that M*y = cross(x,y) for all 3-vectors y.

     Input: 
       - x np.ndarray(3,1) : vector

     Output: 
       - M np.ndarray(3,3) : antisymmetric matrix
    """
    M = np.array([[0,   -x[2], x[1]], 
                  [x[2],  0,  -x[0]],
                  [-x[1], x[0],  0]])
    return M



def distPoint2EpipolarLine(F, p1, p2):
    """ Compute the point-to-epipolar-line distance

       Input:
       - F np.ndarray(3,3): Fundamental matrix
       - p1 np.ndarray(3,N): homogeneous coords of the observed points in image 1
       - p2 np.ndarray(3,N): homogeneous coords of the observed points in image 2

       Output:
       - cost: sum of squared distance from points to epipolar lines
               normalized by the number of point coordinates
    """

    N = p1.shape[1]

    homog_points = np.c_[p1, p2]
    epi_lines = np.c_[F.T @ p2, F @ p1]

    denom = epi_lines[0,:]**2 + epi_lines[1,:]**2
    cost = np.sqrt( np.sum( np.sum( epi_lines * homog_points, axis = 0)**2 / denom) / N)

    return cost

def Matrix2Cross(M):
    """
    Computes the 3D vector x corresponding to an antisymmetric matrix M such that M*y = cross(x,y)
    for all 3D vectors y.
    Input:
     - M(3,3) : antisymmetric matrix
    Output:
     - x(3,1) : column vector
    See also CROSS2MATRIX
    """
    x = np.array([-M[1, 2], M[0, 2], -M[0, 1]])

    return x

def HomogMatrix2twist(H):
    """
    HomogMatrix2twist Convert 4x4 homogeneous matrix to twist coordinates
    Input:
     -H(4,4): Euclidean transformation matrix (rigid body motion)
    Output:
     -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]

    Observe that the same H might be represented by different twist vectors
    Here, twist(4:6) is a rotation vector with norm in [0,pi]
    """

    se_matrix = logm(H)

    # careful for rotations of pi; the top 3x3 submatrix of the returned se_matrix by logm is not
    # skew-symmetric (bad).

    v = se_matrix[:3, 3]

    w = Matrix2Cross(se_matrix[:3, :3])
    twist = np.concatenate([v, w])

    return twist

def twist2HomogMatrix(twist):
    """
    twist2HomogMatrix Convert twist coordinates to 4x4 homogeneous matrix
    Input: -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]
    Output: -H(4,4): Euclidean transformation matrix (rigid body motion)
    """
    v = twist[:3]  # linear part
    w = twist[3:]   # angular part

    se_matrix = np.concatenate([cross2Matrix(w), v[:, None]], axis=1)
    se_matrix = np.concatenate([se_matrix, np.zeros([1, 4])], axis=0)

    H = expm(se_matrix)

    return H