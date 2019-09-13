import csv
import numpy as np

def load_dataset():
    with open('dataset/iris.data', 'rb') as file:
        lines = file.readlines()

    dataset = [[]]

    for i in range(len(lines)):

        new_line = str(lines[i]).split(',')

        features = new_line[:-1]
        class_label = new_line[-1]

        class_label = class_label.split('-')[1][:-3]
        
        # setosa = 0, versicolour = 1, virginica = 2
        if class_label == 'setosa':
            class_label = 0

        elif class_label == 'versicolour':
            class_label = 1
        
        else:
            class_label = 2

        dataset.append([features, class_label])
        #dataset[i].append(class_label)

    return dataset

# testing
dataset = load_dataset()

for i in range(len(dataset)):
    for j in range(len(dataset[0])):
        print(dataset[i][j])

def generate_confusion_matrix(predicted_class, actual_class):
    confusion_matrix = pd.df()

    return confusion_matrix