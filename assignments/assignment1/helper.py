import csv
import numpy as np
import random

def load_dataset():
    with open('iris.data', 'r') as file:
        lines = csv.reader(file)

        dataset = list(lines)[:-1]
        
    for row in dataset:
        class_label = row[4][5:]

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

#training_set, test_set = train_test_split(dataset, 0.7)

#print('Training set size: ', len(training_set))
#print('Test set size: ', len(test_set))

#print(generate_k_folds(dataset))