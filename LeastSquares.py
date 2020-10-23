import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.inv(np.matmul(np.transpose(tx), tx))
    w = np.matmul(w,np.transpose(tx))
    w = np.matmul(w,y)
    return w