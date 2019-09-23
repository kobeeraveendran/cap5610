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

def generate_k_folds(dataset, k = 5):

    random.shuffle(dataset)

    avg, mod = divmod(len(dataset), k)

    folds = list(dataset[i * avg + min(i, mod):(i + 1) * avg + min(i + 1, mod)] for i in range(k))

    return folds

def train_test_split(dataset, split_ratio):
    training_set = []
    test_set = []

    random.shuffle(dataset)
    separation_index = int(split_ratio * len(dataset))

    #print(separation_index)

    training_set = dataset[:separation_index]
    test_set = dataset[separation_index:]

    return training_set, test_set

def generate_confusion_matrix(actual, predicted, plot_title, num_classes = 3, filename = None):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype = int)

    for i in range(len(actual)):
        confusion_matrix[actual[i]][predicted[i]] += 1

    print('\n\n' + plot_title)
    print(confusion_matrix)

    # optional heatmap for better visual idea
    
    # ask if seaborn is allowed
    #plt.figure(figsize = (10, 7))
    #sns.set(font_scale = 1.4)
    #sns.heatmap(pd.DataFrame(confusion_matrix), annot = True)
    #plt.show()

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

#training_set, test_set = train_test_split(dataset, 0.7)

#print('Training set size: ', len(training_set))
#print('Test set size: ', len(test_set))

#print(generate_k_folds(dataset))

#plot = generate_confusion_matrix([2, 0, 2, 2, 0, 1], [0, 0, 2, 2, 0, 2], 'Test confusion matrix', filename = 'test_confusion_matrix.png')

#plt.show(plot)