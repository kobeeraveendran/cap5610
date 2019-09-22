import numpy as np
import argparse
import itertools
import pandas as pd

from helper import load_dataset, generate_k_folds

dataset = load_dataset()

def naive_bayes(dataset):

    folds = generate_k_folds(dataset, k = 5)

    accuracies = []

    for index, fold in enumerate(folds):

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

naive_bayes(dataset)