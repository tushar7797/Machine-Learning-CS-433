import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    #Calculate the loss. You can calculate the loss using mse or mae
    e = (y-tx.dot(w))**2
    #print(y.shape, tx.dot(w))
    return np.mean(e)
    raise NotImplementedError
    
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x
    
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    indices = []
    for j in range(len(x)):
        indices.append(j)
    np.random.shuffle(indices)
    num_training_instances = int(ratio*len(x))
    train_indices = indices[:num_training_instances]
    test_indices = indices[num_training_instances:]
    
    # split the actual data
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return x_train, x_test, y_train, y_test
    
    raise NotImplementedError
    
def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    loss = np.zeros((len(w0),len(w1)))
    min_loss = np.inf
    w_0 = 0
    w_1 = 0
    for i in range(len(w0)):
        for j in range(len(w1)):
            loss[i][j] = compute_loss(y,tx,[w0[i],w1[j]])
            if loss[i][j] < min_loss:
                min_loss = loss[i][j]
                w_0 = w0[i]
                w_1 = w1[i]
    #raise NotImplementedError
    return loss

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    
    a = np.zeros((len(x),degree))
    for i in range(0,len(x)):
        for j in range(0,degree):
            a[i][j] = x[i]**j
            
    return a
    
    #raise NotImplementedError

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and error vector
    # ***************************************************
    e = (y-tx.dot(w))
    return np.transpose(tx).dot(e)/len(y)
    raise NotImplementedError
    
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        loss = compute_loss(y,tx, w)
        grad = compute_gradient(y,tx,w)
        
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w + gamma*compute_gradient(y,tx,w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    e = (y-tx.dot(w))
    return np.transpose(tx).dot(e)/len(y)


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        for minibatch_y, minibatch_tx in batch_iter(y, tx, len(y)):
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            grad = compute_gradient(minibatch_y, minibatch_tx,w) 
            w = w + gamma*compute_gradient(y,tx,w)
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    #raise NotImplementedError
    return losses, ws

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

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    w = np.linalg.inv(np.matmul(np.transpose(tx), tx)+ lambda_*np.identity(np.shape(tx)[1]))
    w = np.matmul(w,np.transpose(tx))
    w = np.matmul(w,y)
    return w
    raise NotImplementedError