import numpy as np
import argparse
import itertools

from helper import load_dataset, generate_k_folds, generate_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('-k', type = int, nargs = 1)
parser.add_argument('-m', type = str, nargs = 1)

args = parser.parse_args()

k = args.k[0]
mode = args.m[0].lower()

dataset = load_dataset()

def knn_iterative(k, distance_metric = 'euclidean'):

    folds = generate_k_folds(dataset, 5)

    print('K: ', k)
    print('Distance metric: ', distance_metric)

    accuracies = []

    for i, fold in enumerate(folds):

        test_set = fold
        training_set = list(itertools.chain.from_iterable(folds[:i] + folds[i + 1:]))

        predictions = []
        actual = []

        correct = 0

        for j in range(len(test_set)):

            nearest_neighbors = []

            for l in range(len(training_set)):
                
                if distance_metric == 'euclidean':
                    
                    nearest_neighbors.append((euclidean_distance(training_set[l][:-1], test_set[j][:-1]), training_set[l][-1]))

                else:
                    nearest_neighbors.append((cosine_distance(training_set[l][:-1], test_set[j][:-1]), training_set[l][-1]))

            nearest_neighbors.sort(key = lambda x: x[0])

            nearest_neighbors = nearest_neighbors[:k]

            classes = {}

            for neighbor in nearest_neighbors:
                if neighbor[-1] in classes:
                    classes[neighbor[-1]] += 1

                else:
                    classes[neighbor[-1]] = 1

            prediction = sorted(classes.items(),  key = lambda x: x[1], reverse = True)[0][0]

            if prediction == test_set[j][-1]:
                correct += 1
                
            predictions.append(prediction)
            actual.append(test_set[j][-1])

        cm = generate_confusion_matrix(
            actual, 
            predictions, 
            'KNN k = {} dist_metric = {} fold {}'.format(k, distance_metric, i + 1), 
            num_classes = 3, 
            filename = 'knn_k{}_metric_{}_fold{}'.format(k, distance_metric, i + 1)
            )

        accuracy = correct / len(test_set)

        print('Current fold accuracy (%): ', accuracy * 100.)

        accuracies.append(accuracy)

    return sum(accuracies) / len(folds)
            


def euclidean_distance(p1, p2):

    p1 = np.array(p1)
    p2 = np.array(p2)

    dist = np.sqrt(np.sum((p1 - p2) ** 2))

    return dist


def cosine_distance(p1, p2):

    p1 = np.array(p1)
    p2 = np.array(p2)

    dist = 1 - (np.dot(p1, p2) / (np.sqrt(np.dot(p1, p1)) * np.sqrt(np.dot(p2, p2))))

    return dist


accuracy = knn_iterative(k, distance_metric = mode)

print('\n\n\nK Nearest Neighbors Average Accuracy (%): ', accuracy * 100)