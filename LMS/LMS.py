#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/9 14:22
# @Author  : Xuqh
# @Site    : 
# @File    : LMS.py
# @Software: PyCharm
import csv
import os

import numpy as np
import pandas as pd
from matplotlib import animation
from nltk.sentiment.util import split_train_test
from numpy import arange
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


def gen_data(theta):
    """
     generate random linear-seperatable data
     1*x[0]+2*x[1]+3*x[2]
    :param theta: array of parameter
    :return:(x,label)
    """
    # X = pd.DataFrame(np.ndarray(shape=(100, 3), dtype=float), columns=['X0', 'X1', 'X2'])
    # X = pd.DataFrame(np.ndarray(shape=(100, 3), dtype=float), columns=['X0', 'X1', 'X2'])
    X = np.random.rand(100, 3)
    X[:, 0] = 1
    label = np.array([1 if (x[0] * theta[0] + x[1] * theta[1] + x[2] * theta[2]) >= 0 else -1 for x in X])
    df_data = pd.DataFrame(X, columns=["X0", "X1", "X2"])
    df_data["label"] = pd.Series(label)
    df_data.to_csv("data.csv")

    return df_data

    return X, label


def hypothesis(x, theta):
    """
    hypothesis funciton in machine learning
    transpose(x)*theta
    :param x: input feature
    :param y: output label
    :return: hypothesis value
    """
    return np.transpose(x).dot(theta)


def train(theta, alpha, x_train, y_train, stoch=True):
    """
    train model
    :param stoch: logical variable for stochastic gradient decent or batch gradient decent
    :param x_train: dataframe of input features
    :param y_train: array of labels
    :return: array of trained theta
    """
    # TODO stochastic gradient descent
    sample_size = len(x_train)
    thetas = [theta]
    while len(thetas) == 1 or np.array_equal(thetas[-1], thetas[-2]):
        for i in range(sample_size):
            # print type(np.array(x_train[i]))
            hx = hypothesis(np.array(x_train[i]), thetas[-1])
            theta = theta + (y_train[i] - hx) * alpha * np.array(x_train[i])
            thetas.append(theta)
            # TODO batch gradient descent
    return theta, thetas


def plot_history(thetas, x):
    """
    plot train process
    :param thetas: array of theta generated in the process
    :param x: array of input features
    :return: void
    """
    pass


#  generate data
theta = np.array([0.22, 0.12, -0.5])
if not os.path.exists("data.csv"):
    gen_data(theta)
data = pd.DataFrame.from_csv("data.csv")
train_X, test_X, train_label, test_label = train_test_split(data.iloc[:, :-1].values, data.iloc[:, -1].values,
                                                            test_size=0.1)
# plot points
fig, ax = plt.subplots()
ax.scatter(train_X[train_label == -1][:, 1], train_X[train_label == -1][:, 2], marker='x', c="red")
ax.scatter(train_X[train_label == 1][:, 1], train_X[train_label == 1][:, 2], marker="o", c="green")
x1 = arange(0, 1, step=0.0001)
x2 = -1 * (theta[0] + theta[1] * x1) / theta[2]
line, = ax.plot(x1, x2)


def init_plot():
    """
    init function for plot animation
    display of all points and seperate line

    :return:line
    """
    x1 = arange(0, 1, step=0.0001)
    x2 = -1 * (theta[0] + theta[1] * x1) / theta[2]
    line.set_ydata(x2)
    return line,


def ani_GDA(theta):
    """
    animation of gradient descent
    :param theta: history data of theta in the progress
    :return: line
    """
    x1 = arange(0, 1, step=0.0001)
    x2 = -1 * (theta[0] + theta[1] * x1) / theta[2]
    line.set_ydata(x2)
    return line,


#  train and save history data
theta, thetas = train(np.array([0, 0, 0]), alpha=0.1, x_train=train_X, y_train=np.array(train_label))
slices = [_ for (i, _) in enumerate(thetas) if i % 2 == 0]
print(len(slices))
ani = animation.FuncAnimation(fig, ani_GDA, slices, interval=250,blit=True,repeat=False)
print (len(thetas))
# TODO plot history data
plt.show()
