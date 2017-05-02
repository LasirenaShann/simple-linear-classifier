import numpy as np

class LogisticRegressor(object):

    def __init__(self, max_iter, verbose, random_state=3):
        """
        Param
            max_iter: maximum iterations of Newton-Raphson
            verbose: if 1, print training loss each iteration
            random_state: random seed
        """
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, t, stop_criteria=100):
        """
        Train the model given input X and target t
        """
        N, M = X.shape
        self.N_feature = 1 + M # including a bias term
        N_class = t.shape[1]
        self.cost = []
        rng = np.random.RandomState(self.random_state)
        bias = np.ones((N,1))
        phi = np.concatenate((bias, X), axis=1) # add a bias term to the feature vector
        # Initialize the model weights randomly
        self.w = rng.normal(loc=0.0, scale=0.01, size=(1 + M, N_class))
        # Newton-Raphson method to optimize model weights
        for i in range(self.max_iter):
            y = self.predict(phi)
            error = y - t
            grad = phi.T.dot(error) / N
            hessian = self.hessian_(phi, y)
            cross_entropy = -((t * np.log(y) + (1 - t) * np.log(1 - y) * (1 - y)).sum()) / N
            self.cost.append(cross_entropy)
            self.w -= np.dot(np.linalg.inv(hessian), grad)
            if error.sum() > stop_criteria:
                print("Error")
                break
            if self.verbose:
                print("[{}] loss: {}".format(i, cross_entropy))


    def predict(self, phi):
        """
        Predict the output value by applying the current weights and softmax function
        """
        if phi.shape[1] < self.N_feature:
            bias = np.ones((phi.shape[0], 1))
            phi = np.concatenate((bias, phi), axis=1)
        y = np.dot(phi, self.w)
        return self.softmax_(y)

    def hessian_(self, phi, y):
        """
        Compute the hessian matrix given phi and y
        Param
            phi: feature matrix, N-by-M
            y: predicted output
        Return
            hessian_matrix
        """
        N_sample, self.N_feature = phi.shape
        N_class = y.shape[1]
        R = np.identity(N_sample)
        for i in range(N_sample):
            h = np.dot(y[i,:].T, 1-y[i,:])
            R[i,i] = h
        hessian_matrix = np.dot(phi.T, np.dot(R, phi))
        return hessian_matrix

    def softmax_(self, x):
        """
        softmax function
        """
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=1)[:,None]

    def encode_one_hot(self, y):
        """
        Encode the softmax output into one-hot vector
        Param
            y: output of softmax, N-by-K matrix
        Return
            one_hot: one-hot encoded, N-by-K matrix
        """
        labels = np.identity(y.shape[1])
        one_hot = labels[np.argmax(y, axis=1)]
        return one_hot

