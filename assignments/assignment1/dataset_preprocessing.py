import csv
import numpy as np

def load_dataset():
    with open('dataset/iris.data', 'r') as file:
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

# testing
dataset = load_dataset()

import random

def train_test_split(dataset, split_ratio):
    training_set = []
    test_set = []

    random.shuffle(dataset)
    separation_index = int(split_ratio * len(dataset))

    #print(separation_index)

    training_set = dataset[:separation_index]
    test_set = dataset[separation_index:]

    return training_set, test_set

training_set, test_set = train_test_split(dataset, 0.7)


print('Training set')
print('size: ', len(training_set))
for row in training_set:
    print(row)

print('Test set')
print('size: ', len(test_set))
for row in test_set:
    print(row)