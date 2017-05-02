import numpy as np
from dim_reduction import *
from util import *

class QuadraticClassifier(object):
    """
    A classifier model by Quadratic Discriminant Analysis(QDA)
    """
    def __init__(self):
        pass

    def fit(self, X, t):
        """
        Fit the model given X and t
        """
        (N, N_class) = t.shape
        M = X.shape[1]
        self.labels = np.identity(N_class) # one-hot class labels
        self.w = np.zeros((M+1, N_class))
        for i in range(N_class):
            idx =  np.where(np.all(t==self.labels[i], axis=1))[0]
            C = X[idx]
            mu, sigma = gaussianMLE(C)
            prior = C.shape[0] / float(N)
            self.w[0, i] = -0.5 * np.dot(mu.T, np.dot(np.linalg.inv(sigma), mu)) + np.log(prior) # bias
            self.w[1:,i] = np.dot(np.linalg.inv(sigma), mu) # weights

    def predict(self, X):
        y = np.dot(X, self.w[1:,:]) + self.w[0,:]
        y = self.softmax_(y)
        predictions = self.labels[np.argmax(y, axis=1)] # encode it in one-hot vectors
        return predictions

    def softmax_(self, x):
        """
        softmax function
        """
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=1)[:,None]

class NaiveBayesClassifier(object):

    def __init__(self):
        self.priors = []

    def fit(self, X, t):
        (N, N_feature) = X.shape
        self.N_class = t.shape[1]
        #t_bar = np.ascontiguousarray(t).view(np.dtype((np.void, t.dtype.itemsize * t.shape[1])))
        #_, idx = np.unique(t_bar, return_index=True)
        self.labels = np.identity(self.N_class)#t[idx]
        self.mus = np.zeros((self.N_class, N_feature))
        self.sigmas = np.zeros((self.N_class, N_feature))
        for i in range(self.N_class):
            idx =  np.where(np.all(t==self.labels[i], axis=1))
            C = X[idx]
            for j in range(N_feature):
                (mu, sigma) = gaussianMLE(C[:,j])
                #print sigma
                self.mus[i,j] = mu
                self.sigmas[i,j] = sigma
            #print self.sigmas
            self.priors.append(C.shape[0] / float(N))

    def predict(self, X):
        (N, dim) = X.shape
        likelihood = np.zeros((N, self.N_class))
        for i in range(self.N_class):
            likelihood[:, i] = np.log(self.priors[i])
            for j in range(dim):
                likelihood[:, i] = likelihood[:, i] - dim * 0.5 * (np.log(2*np.pi) + np.log(self.sigmas[i, j]))
                X_bar = X[:,j] - self.mus[i, j]
                likelihood[:, i] = likelihood[:, i] - 0.5 * np.sum(np.dot(np.dot(X_bar, 1.0/self.sigmas[i, j]), X_bar.T), axis=0)
        ak = np.dot(X, np.log(self.mus).T) + np.dot((1-X), np.log(1-self.mus).T) + np.log(self.priors)
        ak = self.softmax_(ak)
        idx = np.argmax(ak, axis=1)
        predictions = self.labels[idx]
        return predictions

    def softmax_(self, x):
        """
        softmax function
        """
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=1)[:,None]

