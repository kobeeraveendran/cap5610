import csv
import numpy as np

def load_dataset():
    with open('iris.data', 'rb') as file:
        lines = file.readlines()

    dataset = [[]]

    for i in range(len(lines)):
        new_line = lines[i].split(',')

        features = new_line[:-1]
        class_label = new_line[-1]

        class_label = class_label.split('-')[1]
        
        # setosa = 0, versicolour = 1, virginica = 2
        if class_label == 'setosa':
            class_label = 0

        elif class_label == 'versicolour':
            class_label = 1
        
        else:
            class_label = 2

        dataset[i] = features
        dataset[i].append(class_label)

    return dataset