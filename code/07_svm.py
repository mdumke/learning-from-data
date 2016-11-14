#!/home/tias/anaconda3/bin/python3

"""
Compare the performance of perceptrons and support vector machines
on self-generated, linearly separable data
"""

import sys
import os
import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt.solvers import qp
import matplotlib.pyplot as plt


class DataGenerator():
    """
    Generates target functions and training/test data
    """
    def __init__(self):
        self.target = self.generate_target()

    def generate_target(self):
        """
        returns a function that classifies a given point as +/- 1 along
        a linear decision boundary
        """
        x1 = np.random.random(2) * 2 - 1
        x2 = np.random.random(2) * 2 - 1

        m = (x1[1] - x2[1]) / (x1[0] - x2[0])
        b = x1[1] - m * x1[0]

        def target(x):
            return  1 if (m * x[0] + b < x[1]) else -1

        return target

    def generate_data(self, n, target=None):
        """
        returns n data points (xs and ys) generated from the given function
        """
        target = target if target else self.target

        while True:
            x = np.random.random(n * 2).reshape(n, 2) * 2 - 1
            y = self.predict(x, target)

            # make sure there are points on both sides of the line
            if abs(y.sum()) != n: break

        return x, y

    def predict(self, data, function):
        """
        returns the function's predicted values on the data
        """
        return np.apply_along_axis(function, 1, data)


class Evaluator():
    """
    estimates in- and out-of sample errors for different
    datasets and functions
    """
    def compute_misclassification_rate(self, target, hypothesis, n = 1000):
        """
        returns an estimate for the number of different predictions
        using n points
        """
        generator = DataGenerator()

        x_test, y_test = generator.generate_data(n, target)
        predictions = generator.predict(x_test, hypothesis)

        num_mismatches = (np.abs((y_test - predictions)) / 2).sum()
        return num_mismatches / n


class PLA():
    """
    Implements the perceptron learning algorithm
    """
    def train(self, x, y):
        """
        returns a classification function learned from the given datapoints
        """
        # no risk: make sure data is not mutated
        x = np.copy(x)
        y = np.copy(y)

        w = np.zeros(3)
        x = self.add_bias_column(x)
        n = x.shape[0]

        done = False

        while (not done):
            done = True
            x, y = self.random_permutation(x, y)

            ## perform weights-update on the first misclassified point
            for i in range(n):
                if np.sign(w.dot(x[i])) != y[i]:
                    w = w + y[i] * x[i]
                    done = False
                    break

        return self.as_classification_function(w)


    def as_classification_function(self, w):
        """
        turns the given weights-vector into a +/-1 classification-function
        """
        def hypothesis(point):
            prediction = w[0] + w[1] * point[0] + w[2] * point[1]
            return 1 if np.sign(prediction) == 1 else -1

        return hypothesis


    def random_permutation(self, x, y):
        """
        returns the data with the same permutation applied to both parts
        """
        permutation = np.random.permutation(x.shape[0])

        x = pd.DataFrame(x.T)[permutation].as_matrix().T
        y = y[permutation]

        return x, y


    def add_bias_column(self, data):
        """
        prepends the 2-dim data-array with a column of 1s
        """
        return pd.DataFrame({
            0: np.ones(x.shape[0]),
            1: x[:, 0],
            2: x[:, 1]
        }).as_matrix()


class SVM():
    """
    Implements hard-margin support vector machines
    """
    def train(self, x, y):
        """
        returns a classification function learned from the given data points
        """
        n = x.shape[0]
        quadtratic_coefficients = np.zeros(n * n).reshape(n, n)

        for i in range(n):
            for j in range(n):
                quadtratic_coefficients[i][j] = y[i] * y[j] * x[i].dot(x[j])

        # pass the coefficients to quadratic programming
        P = matrix(quadtratic_coefficients)
        q = matrix(np.zeros(n) - 1)
        G = matrix(np.eye(n) * -1)
        h = matrix(0.0, (n ,1))
        A = matrix(y * 1.0, (1, n))
        b = matrix(0.0)

        # prevent long output
        sys.stdout = open(os.devnull, "w")
        alpha = np.array(qp(P, q, G, h, A, b)['x'])
        sys.stdout = sys.__stdout__

        # compute the weights (w/o the bias)
        w = np.zeros(2)

        for i in range(n):
            w += alpha[i] * y[i] * x[i]

        # compute the bias
        x_help = None
        y_help = None

        zero-threshold = 1e-5

        for i in range(n):
            if alpha[i] > zero-threshold:
                x_help = x[i]
                y_help = y[i]

        bias = (1 - y_help * w.dot(x_help)) / y_help

        num_support_vectors = (alpha > zero-threshold).sum()

        # turn the results into a hypothesis-function
        def hypothesis(point):
            prediction = bias + w[0] * point[0] + w[1] * point[1]
            return 1 if np.sign(prediction) == 1 else -1

        return hypothesis, num_support_vectors



num_trials = 500
num_data_points = 100

pla_error_rate = np.zeros(num_trials)
svm_error_rate = np.zeros(num_trials)
num_vectors = np.zeros(num_trials)

generator = DataGenerator()
evaluator = Evaluator()
pla = PLA()
svm = SVM()

for i in range(num_trials):
    if i % 50 == 0: print(i)

    target = generator.generate_target()
    x, y = generator.generate_data(num_data_points, target)

    g_pla = pla.train(x, y)
    g_svm, num_vectors[i] = svm.train(x, y)

    pla_error_rate[i] = evaluator.compute_misclassification_rate(target, g_pla)
    svm_error_rate[i] = evaluator.compute_misclassification_rate(target, g_svm)

print('svm wins:', (svm_error_rate < pla_error_rate).sum() / num_trials)
print('avg # SVs:', num_vectors.sum() / num_trials)


