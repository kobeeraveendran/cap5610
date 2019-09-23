# Kobee Raveendran
# University of Central Florida
# CAP5610 Assignment 1 - Fall 2019
# 9/21/2019

import numpy as np
import argparse
import itertools
import pandas as pd

from helper import load_dataset, generate_k_folds, generate_confusion_matrix

dataset = load_dataset()

# primary naive bayes function:
# inputs: full dataset, number of class categories
# outputs: average accuracy over all 5 CV folds (for each test set)
# desc: fits a Gaussian Naive Bayes classifier on training dataset
def naive_bayes(dataset, num_classes):

    folds = generate_k_folds(dataset, k = 5)

    accuracies = []

    for index, fold in enumerate(folds):

        test_set = fold
        training_set = list(itertools.chain.from_iterable(folds[:index] + folds[index + 1:]))

        test_size = len(test_set)
        training_size = len(training_set)

        test_set = pd.DataFrame(test_set)
        training_set = pd.DataFrame(training_set)

        X_train = training_set.iloc[:, 0:4]
        Y_train = training_set.iloc[:, 4]

        X_test = test_set.iloc[:, 0:4]
        Y_test = test_set.iloc[:, 4]
        

        num_classes = 3

        classes = []

        # separate the dataset by class

        for class_index in range(num_classes):
            class_splits = training_set[training_set[4] == class_index].iloc[:, 0:4]
            classes.append(class_splits)


        # compute and store feature-wise means and standard deviations for each class
        means = []
        std_devs = []

        for category in classes:
            means.append(np.mean(category, axis = 0))
            std_devs.append(np.std(category, axis = 0))

        predictions = []

        for i in range(len(X_test)):
            pred = predict(means, std_devs, X_test.iloc[i, :])
            predictions.append(pred)

        actual = Y_test.values.tolist()

        correct = 0

        for i in range(len(X_test)):
            if Y_test[i] == predictions[i]:
                correct += 1

        accuracy = correct / len(Y_test)

        accuracies.append(accuracy)

        cm = generate_confusion_matrix(
            actual, 
            predictions, 
            plot_title = 'Naive Bayes fold {}'.format(index + 1), 
            num_classes = 3, 
            filename = 'naive_bayes_fold{}'.format(index + 1)
        )


        print('Current fold accuracy (%): ', accuracy * 100.)

    return sum(accuracies) / len(accuracies)

# Gaussian probability distribution function
def gaussian_likelihood(X, mean, std):
    
    # ask about variants of this formula (i.e. whether std should be squared in first part, under sqrt, etc.)
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-1 * (X - mean) ** 2 / (2 * std ** 2))
    
# calculates the probabilities of a sample belonging to each class
def class_probabilities(means, std_devs, input_vector):

    probabilites = {}

    for class_index in range(len(means)):
        for i in range(len(means[0])):
            mean = means[class_index][i]
            std_dev = std_devs[class_index][i]

            x = input_vector[i]

            probabilites[class_index] = gaussian_likelihood(x, mean, std_dev)

    return probabilites

# given an input vector of features, associates with the most likely class
# to which it may belong
def predict(means, std_devs, input_vector):
    probs = class_probabilities(means, std_devs, input_vector)

    best_label = None
    max_prob = -1

    for class_index, prob in probs.items():
        if best_label is None or prob > max_prob:
            best_label = class_index
            max_prob = prob

    return best_label

accuracy = naive_bayes(dataset, num_classes = 3)

print('\n\nNaive Bayes Average accuracy (%): ', accuracy * 100.)