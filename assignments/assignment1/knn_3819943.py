import numpy as np
import math
import argparse
import itertools

from helper import load_dataset, generate_k_folds

parser = argparse.ArgumentParser()
parser.add_argument('-k', type = int, nargs = 1)
parser.add_argument('-m', type = str, nargs = 1)

args = parser.parse_args()

k = args.k[0]
mode = args.m[0].lower()

print(mode)

dataset = load_dataset()

def k_nearest_neighbors(k = 5, distance_metric = 'euclidean'):
    
    folds = generate_k_folds(dataset, k = 5)

    for fold in folds:
        test_set = folds.pop(fold)

        training_set = list(itertools.chain.from_iterable(folds))

        if distance_metric == 'euclidean':
            dists = euclidean_distance(test_set, training_set)

        else:
            dists = cosine_distance(test_set, training_set)

def euclidean_distance(p1, p2):

    return np.sqrt(np.sum((p1 - p2) ** 2))


def cosine_distance(p1, p2):
    
    return 1 - (np.dot(p1, p2) / (np.sqrt(np.dot(p1, p1)) * np.sqrt(np.dot(p2, p2))))