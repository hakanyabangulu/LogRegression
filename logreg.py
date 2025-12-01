import numpy as np

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        
    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        hres = self.sigmoid(X @ theta) # h(x) = g(theta*x)
        regres = ((regLambda / 2) * np.sum(theta[1:]**2)) / len(y) # ((Lambda / 2) * theta(1 to d)^2)  / n
        jreg = (((-1) * (np.sum(y*np.log(hres)) + (1-y) * np.log(1-hres))) / len(y))  + regres # (-1) * (y*log(h) + (1-y)*log(1-h)) 
        return jreg
        
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        hres = self.sigmoid(X @ theta) # h(x) = g(theta*x)
        gradres = (X.T @ (hres - y)) / len(y)  # (h(x) - y) * x)
        gradres[1:] = gradres[1:] + regLambda * theta[1:] # (h(x) - y) * x) + lambda*theta 
        return gradres
    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        ** the d here is different from above! (due to augmentation) **
        '''
        
        n, d = X.shape # For bias.
        X = np.c_[np.ones((n, 1)), X]  # n x (d+1)
        self.theta = np.random.randn(d + 1) * 0.01  # Random value with 0 mean.
        
        
        for _ in range(self.maxNumIters): # Gradient descent
            current_cost = self.computeCost(self.theta, X, y, self.regLambda)
            gradres = self.computeGradient(self.theta, X, y, self.regLambda)
            
            theta_old = self.theta.copy() # Updating theta.
            self.theta = self.theta - self.alpha * gradres
            
            theta_diff = self.theta - theta_old  #||θ_new - θ_old||
            if np.linalg.norm(theta_diff, 2) < self.epsilon: #||θ_new - θ_old||_2 ≤ ε
                break

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions, the output should be binary (use h_theta > .5)
        '''
      
        n, d = X.shape # For bias
        X = np.c_[np.ones((n, 1)), X]  # n x (d+1) 
        prob = self.sigmoid(X @ self.theta) # Calculating prob.
        pred = (prob > 0.5).astype(int) # if theta > 0.5 then 1 , else 0.
        return pred

    def sigmoid(self, Z):
        '''
        Applies the sigmoid function on every element of Z
        Arguments:
            Z can be a (n,) vector or (n , m) matrix
        Returns:
            A vector/matrix, same shape with Z, that has the sigmoid function applied elementwise
        '''
       
        res = 1 / (1 + np.exp(-Z))  # g(z) = 1 / (1 + e^(-z))
        self.result = res
        return res