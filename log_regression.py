import numpy as np

def sigmoid(t):
    """apply the sigmoid function on t."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    return 1/(1+np.exp(-t))
    raise NotImplementedError
    
def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    loss = 0
    for i in range(len(y)):
        loss = loss + y[i]*np.log(sigmoid(np.matmul(tx[i],w))) + (1-y[i])*np.log(1-sigmoid(np.matmul(tx[i],w)))
        
    loss = loss*-1/len(y)
    return loss
    
    raise NotImplementedError
    
    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    grad = sigmoid(np.matmul(tx,w)) - y
    grad = np.matmul(np.transpose(tx),grad)
    return -grad
    
    raise NotImplementedError
    
def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # compute the loss: TODO
    # ***************************************************
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    w = w + grad*gamma
    
    #raise NotImplementedError
    # ***************************************************
    # INSERT YOUR CODE HERE
    # compute the gradient: TODO
    # ***************************************************
    
    #raise NotImplementedError
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    # ***************************************************
    #raise NotImplementedError
    
    return loss, w


