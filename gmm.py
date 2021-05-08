# -*- coding: utf-8 -*-
"""
Created on Wed May  5 20:38:10 2021

@author: wanghaohui
"""

import numpy as np
from loggausspdf import chol_loggausspdf
from scipy.special import logsumexp


class GMM(object):
    def __init__(self, K = 3, tol = 1e-5, maxiter = 100):
        """
        Args:
            K: int, the number of GMM
            tol: float, the thresholding for stop
            maxiter: int, maximal iteration number

        """
        self.K = K
        self.tol = tol
        self.maxiter = maxiter
        self.loglike = 0 # The log likelihood
        self.reg_covar = 1e-7 # A decimal added to cov
        self.Ktuning_tol = 1e-5 # Threshold of merging Gaussian mixture distribution
        
    def fit(self, trainMat):
        """
        Args:
            trainMat: array, N x D input data matrix, 
                      where N refers to the number of samples and D refers to the number of features

        Returns:   
            a GMM model trained with trainMat, where:
                mu: K × D, the means of the different Gaussians
                cov: K × D × D, the variance of the different Gaussians, sigma^2
                alpha: (K, ), the mixing coefficients, represented by Pi in article
                gamma: N × K, the d-th sample belongs to the k-th Gaussian distribution
                loglike: float, the log likelihood

        """
        self.X = trainMat
        self.N, self.D = trainMat.shape
        self.GMM_EM()
        
    def GMM_EM(self):
        """Perform EM algorithm for fitting the GMM model
        Step 1: Pre-process data;
        Step 2: Initialize parameters;
        Step 3: Expectation;
        Step 4: Maximum.
        
        """
        self.scale_data()
        self.init_params()
        for i in range(self.maxiter):
            log_prob_norm, self.gamma = self.E_Step(self.X)
            self.mu, self.cov, self.alpha = self.M_Step()
            newloglike = self.loglikelihood(log_prob_norm)
            
            if abs(newloglike - self.loglike) < self.tol:
                print("Stop early in %d-th iteration" % i)
                break
            self.loglike = newloglike
        print("{sep} Result {sep}".format(sep="-" * 20))
        print("mu:", self.mu, "cov:", self.cov, "alpha:", self.alpha, sep="\n")
    
    def scale_data(self):
        """Pre-process data, scale data between 0 and 1
        
        """
        for d in range(self.D):
            max_ = self.X[:, d].max()
            min_ = self.X[:, d].min()
            self.X[:, d] = (self.X[:, d] - min_) / (max_ - min_)
        print("Data scaled.")
        self.xj_mean = np.mean(self.X, axis=0) # Mean value of each column (feature)
        self.xj_s = np.sqrt(np.var(self.X, axis=0)) # Standard deviation each column (feature)

    def init_params(self):
        """Initialize parameters
        
        """
        self.mu = np.random.rand(self.K, self.D) 
        self.cov = np.array([np.eye(self.D)] * self.K) * 0.1
        self.alpha = np.array([1.0 / self.K] * self.K) 
        print("Parameters initialized.")
        print("mu:", self.mu, "cov:", self.cov, "alpha:", self.alpha, sep="\n")
                
    def E_Step(self, data):
        """E step, estimate gamma
        
        Args:
            data: array, N x D input data matrix, one row per sample
            
        Returns:
            log_prob_norm: N × K, log likelihood of each sample
            np.exp(log_gamma): the update of gamma
        
        """
        gamma_log_prob = np.mat(np.zeros((self.N, self.K)))

        for k in range(self.K):
            gamma_log_prob[:, k] = np.log(self.alpha[k]) + chol_loggausspdf(data.T, self.mu[k].reshape((self.D,1)), self.cov[k]).reshape(self.N,1)
        log_prob_norm = logsumexp(gamma_log_prob, axis=1)
        log_gamma = gamma_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, np.exp(log_gamma)

    def M_Step(self):
        """M step, maximun loglikelihood
        
        Returns:
            the update of mu, cov and alpha
        
        """
        newmu = np.zeros([self.K, self.D])
        newcov = []
        newalpha = np.zeros(self.K)
        for k in range(self.K):
            Nk = np.sum(self.gamma[:, k])
            newmu[k, :] = np.dot(self.gamma[:, k].T, self.X) / Nk
            cov_k = self.compute_cov(k, Nk)
            newcov.append(cov_k)
            newalpha[k] = Nk / self.N

        newcov = np.array(newcov)
        return newmu, newcov, newalpha
    
    def compute_cov(self, k, Nk):
        """Calculating cov to prevent non positive definite matrix reg_covar
        
        """
        diff = np.mat(self.X - self.mu[k])
        cov = np.array(diff.T * np.multiply(diff, self.gamma[:,k]) / Nk)
        cov.flat[::self.D + 1] += self.reg_covar
        return cov
    
    def loglikelihood(self, log_prob_norm):
        """Approximation algorithm of log to prevent underflow and overflow
        
        """
        return np.sum(log_prob_norm)

    # def loglikelihood(self):
    #     P = np.zeros([self.N, self.K])
    #     for k in range(self.K):
    #         P[:,k] = prob(self.X, self.mu[k], self.cov[k])
    #
    #     return np.sum(np.log(P.dot(self.alpha)))






