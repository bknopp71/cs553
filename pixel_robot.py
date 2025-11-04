import numpy as np

def pixel_to_fanuc(u, v, M):
    """
    Convert pixel coordinates (u, v) to Fanuc robot coordinates (X, Y)
    using an affine transformation matrix M.

    Parameters:
        u, v : float
            Pixel coordinates from the camera image
        M : numpy.ndarray of shape (2, 3)
            Affine transformation matrix:
                [ [a, b, c],
                  [d, e, f] ]

    Returns:
        X, Y : float
            Robot coordinates (in mm)
    """
    # Convert pixel to homogeneous form [u, v, 1]
    pixel_vec = np.array([u, v, 1.0])

    # Matrix multiply: [X, Y] = M * [u, v, 1]
    X, Y = np.dot(M, pixel_vec)

    return float(X), float(Y)