import numpy as np
import argparse
import itertools
import pandas as pd

from helper import load_dataset, generate_k_folds

dataset = load_dataset()

def naive_bayes(dataset, num_classes):

    folds = generate_k_folds(dataset, k = 5)

    accuracies = []

    for index, fold in enumerate(folds):

        if index == 1:
            break

        test_set = fold
        training_set = list(itertools.chain.from_iterable(folds[:index] + folds[index + 1:]))

        test_size = len(test_set)
        training_size = len(training_set)

        print(test_size)
        print(training_size)

        test_set = pd.DataFrame(test_set)
        training_set = pd.DataFrame(training_set)

        print(test_set)

        # separate by class value
        X_train = training_set.iloc[:, 0:4]
        Y_train = training_set.iloc[:, 4]

        X_test = test_set.iloc[:, 0:4]
        Y_test = test_set.iloc[:, 4]

        #print('X_train')
        #print(X_train)

        #print('\n\n\nY_train')
        #print(Y_train)

        num_classes = 3

        classes = []
        prob_classes = []

        for class_index in range(num_classes):
            class_splits = training_set[training_set[4] == class_index].iloc[:, 0:4]
            prob_classes.append(len(class_splits))
            classes.append(class_splits)
            #classes.append(training_set[training_set[4] == class_index].iloc[:, 0:4])

        print(classes[0])

        print(len(classes))


        # compute and store feature-wise means and standard deviations
        means = []
        std_devs = []

        for category in classes:
            means.append(np.mean(category, axis = 0))
            std_devs.append(np.std(category, axis = 0))

        score_list = []     

        # compute p(y) for each class
        prob_classes = [prob / len(training_set) for prob in prob_classes]

        print('Class probabilities: ')
        print(prob_classes)

        #print('\n\n\nMeans: ', means)
        #print('Standard deviations: ', std_devs)

def gaussian_feature_probability(X, mean, std):
    # check formula
    return (1 / (np.sqrt(2 * np.pi * std ** 2))) * np.exp(-1 * (X - mean) ** 2 / (2 * std ** 2))


naive_bayes(dataset, num_classes = 3)