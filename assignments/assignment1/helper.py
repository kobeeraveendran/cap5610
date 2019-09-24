# Kobee Raveendran
# University of Central Florida
# CAP5610 Assignment 1 - Fall 2019
# 9/13/2019

import csv
import numpy as np
import random
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import os

# loads dataset from 'iris.data' in root project directory
# as a 2-dimensional list of lists
# rows = training + testing examples
# cols = labels
#   > cols 0 - 3: features
#   > col 4: class label, encoded from string to integer
def load_dataset():
    with open('iris.data', 'r') as file:
        lines = csv.reader(file)

        dataset = list(lines)[:-1]
        
    for row in dataset:
        class_label = row[4][5:]

        row[:-1] = list(map(float, row[:-1]))

        if class_label == 'setosa':
            row[4] = 0

        elif class_label == 'versicolor':
            row[4] = 1

        else:
            row[4] = 2

    return dataset

# splits the dataset loaded by the function above 
# into k-equally partitioned sections
# where one partition serves as the test set and 
# all others become the training set
def generate_k_folds(dataset, k = 5):

    random.shuffle(dataset)

    avg, mod = divmod(len(dataset), k)

    folds = list(dataset[i * avg + min(i, mod):(i + 1) * avg + min(i + 1, mod)] for i in range(k))

    return folds


# generates text-based and plot-based confusion matrix
# inputs:
#   - actual: array of the actual class labels for the test set
#   - predicted: array of the predicted class labels for every element of the test set
#   - plot_title: string representing title of the confusion matrix (for matplotlib plot)
#   - num_classes: number of unique classes in dataset
#   - filename: if provided, saves the plot of the confusion matrix locally in folder 'plots'
def generate_confusion_matrix(actual, predicted, plot_title, num_classes = 3, filename = None):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype = int)

    for i in range(len(actual)):
        confusion_matrix[actual[i]][predicted[i]] += 1

    print('\n\n' + plot_title)
    print(confusion_matrix)

    # using matplotlib only
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation = 'nearest', cmap = plt.cm.Blues)
    ax.figure.colorbar(im, ax = ax)
    ax.set(xticks = np.arange(confusion_matrix.shape[0]), 
           yticks = np.arange(confusion_matrix.shape[1]), 
           xticklabels = ['setosa', 'versicolor', 'virginica'], 
           yticklabels = ['setosa', 'versicolor', 'virginica'], 
           xlabel = 'Predicted class', 
           ylabel = 'True class')

    plt.title(plot_title, y = 1.12)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    plt.setp(ax.get_xticklabels(), ha = 'center', rotation_mode = 'anchor')

    threshold = confusion_matrix.max() / 2

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'), ha = 'center', va = 'center', 
            color = 'white' if confusion_matrix[i, j] > threshold else 'black')

    fig.tight_layout()

    if filename:
        if not os.path.isdir('plots'):
            os.makedirs('plots')

        plt.savefig('plots/' + filename)

    return ax

# TODO: maybe move the plotting of the confusion matrix to a standalone function
def average_confusion_matrices(confusion_matrices):

    agg_cm = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    for matrix in confusion_matrices:
        agg_cm = agg_cm + matrix

    agg_cm = np.divide(agg_cm, len(confusion_matrices))

    filename = 'nb_average_cm.png'

    print(agg_cm)

    fig, ax = plt.subplots()
    im = ax.imshow(agg_cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    ax.figure.colorbar(im, ax = ax)
    ax.set(xticks = np.arange(agg_cm.shape[0]), 
           yticks = np.arange(agg_cm.shape[1]), 
           xticklabels = ['setosa', 'versicolor', 'virginica'], 
           yticklabels = ['setosa', 'versicolor', 'virginica'], 
           xlabel = 'Predicted class', 
           ylabel = 'True class')

    plt.title('Avg of Naive Bayes CMs', y = 1.12)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    plt.setp(ax.get_xticklabels(), ha = 'center', rotation_mode = 'anchor')

    threshold = agg_cm.max() / 2

    for i in range(agg_cm.shape[0]):
        for j in range(agg_cm.shape[1]):
            ax.text(j, i, format(agg_cm[i, j], '.2f'), ha = 'center', va = 'center', 
            color = 'white' if agg_cm[i, j] > threshold else 'black')

    fig.tight_layout()

    if filename:
        if not os.path.isdir('plots'):
            os.makedirs('plots')

        plt.savefig('plots/' + filename)

# ignore: I only used this to create the aggregate matrix for the best-performing knn condition
# and one for naive bayes
'''
knn_fold1 = np.matrix([[8, 0, 0], [0, 14, 0], [0, 0, 8]])
knn_fold2 = np.matrix([[9, 0, 0], [0, 10, 0], [0, 0, 11]])
knn_fold3 = np.matrix([[15, 0, 0], [0, 8, 1], [0, 0, 6]])
knn_fold4 = np.matrix([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
knn_fold5 = np.matrix([[8, 0, 0], [0, 5, 2], [0, 1, 14]])
'''
'''
nb_fold1 = np.matrix([[11, 0, 0], [0, 8, 0], [0, 2, 9]])
nb_fold2 = np.matrix([[7, 0, 0], [0, 12, 0], [0, 1, 10]])
nb_fold3 = np.matrix([[12, 0, 0], [0, 8, 1], [0, 1, 8]])
nb_fold4 = np.matrix([[11, 0, 0], [0, 12, 0], [0, 0, 7]])
nb_fold5 = np.matrix([[9, 0, 0], [0, 8, 1], [0, 0, 12]])

average_confusion_matrices([nb_fold1, nb_fold2, nb_fold3, nb_fold4, nb_fold5])
'''
