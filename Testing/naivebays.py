import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn


class GaussianNaiveBayes:
    def fit(self, X, y, spar=10e-3):
        number_of_sample, number_of_features = X.shape

        self.categories = np.unique(y)

        number_of_classes = len(self.categories)

        self.gaussian_mean = np.zeros((number_of_classes, number_of_features), dtype=np.float64)
        self.gaussian_var = np.zeros((number_of_classes, number_of_features), dtype=np.float64)
        self.log_prior = np.zeros(number_of_classes, dtype=np.float64)

        for classes in self.categories:
            X_classes = X[classes == y]
            self.gaussian_mean[classes:] = X_classes.mean(axis=0)
            self.gaussian_var[classes:] = X_classes.var(axis=0) + spar
            self.log_prior[classes] = np.log(X_classes.shape[0] / float(number_of_sample))

    def predict(self, X):
        posteriorS = np.zeros((X.shape[0], len(self.categories)))
        for classes in self.categories:
            posteriorS[:, classes] = mvn.logpdf(X, mean=self.gaussian_mean[classes, :],
                                                cov=self.gaussian_var[classes, :]) + self.log_prior[classes]
        return np.argmax(posteriorS, axis=1)

    def accuracy(self, y_true, predicted):
        return np.mean(y_true == predicted)
