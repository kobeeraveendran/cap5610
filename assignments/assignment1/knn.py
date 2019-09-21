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

def knn_iterative(k = 5, distance_metric = 'euclidean'):

    folds = generate_k_folds(dataset, k = 5)

    accuracies = []

    for i, fold in enumerate(folds):

        test_set = folds
        training_set = folds[:i] + folds[i + 1:]

        dists = np.zeros((len(test_set), len(training_set)))

        correct = 0

        for j in range(len(test_set)):

            nearest_neighbors = []

            for k in range(len(training_set)):
                
                if distance_metric == 'euclidean':
                    nearest_neighbors.append((euclidean_distance(training_set[k][:-1], test_set[j][:-1]), training_set[k][:-1]))

                else:
                    nearest_neighbors.append((cosine_distance(training_set[k][:-1], test_set[j][:-1]), training_set[k][-1]))

            nearest_neighbors.sort(key = lambda x: x[0])

            nearest_neighbors = nearest_neighbors[:k]

            classes = {}

            for neighbor in nearest_neighbors:
                if neighbor[-1] in classes:
                    classes[neighbor[-1]] += 1

                else:
                    classes[neighbor[-1]] = 1

            prediction = sorted(classes.items(),  key = lambda x: x[1], reverse = True)[0]

            if prediction == test_set[j][-1]:
                correct += 1

        accuracy = correct / len(dataset)
        accuracies.append(accuracy)

    return np.sum(accuracies) / len(folds)
            


def euclidean_distance(p1, p2):

    p1 = np.array(p1)
    p2 = np.array(p2)

    return np.sqrt(np.sum((p1 - p2) ** 2))


def cosine_distance(p1, p2):

    p1 = np.array(p1)
    p2 = np.array(p2)
    
    return 1 - (np.dot(p1, p2) / (np.sqrt(np.dot(p1, p1)) * np.sqrt(np.dot(p2, p2))))