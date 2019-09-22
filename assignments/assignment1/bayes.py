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

        # separate the dataset by class

        for class_index in range(num_classes):
            class_splits = training_set[training_set[4] == class_index].iloc[:, 0:4]
            prob_classes.append(len(class_splits))
            classes.append(class_splits)
            #classes.append(training_set[training_set[4] == class_index].iloc[:, 0:4])

        print('Rows with class 0: ')
        print(classes[0])

        print(len(classes))


        # compute and store feature-wise means and standard deviations for each class
        means = []
        std_devs = []

        for category in classes:
            means.append(np.mean(category, axis = 0))
            std_devs.append(np.std(category, axis = 0))  

        print('Mean[0]')
        print(means[0])
        print('Std_dev[0]')
        print(std_devs[0])

        # compute p(y) for each class
        prob_classes = [prob / len(training_set) for prob in prob_classes]

        print('Class probabilities: ')
        print(prob_classes)

        #probabilities_list = []

        #for i in range(num_classes):
        #    probabilities_list.append(posterior(X_test, classes[i], means[i], std_devs[i]))


        '''
        prob = (feature_likelihood(classes[0], means[0], std_devs[0]) * 
                feature_likelihood(classes[1], means[1], std_devs[1]) * 
                feature_likelihood(classes[2], means[2], std_devs[2]) *
                prob_classes[i])
        '''



            #probabilities_list.append(prob)

        #print('Probabilities list: ', probabilities_list)

        #predicted_class = np.argmax(probabilities_list)

        #print(predicted_class)

        #print('\n\n\nMeans: ', means)
        #print('Standard deviations: ', std_devs)

        predictions = []

        for i in range(len(X_test)):
            pred = predict(means, std_devs, X_test.iloc[i, :])
            predictions.append(pred)

        print('Predictions')
        print(predictions)

        correct = 0

        for i in range(len(X_test)):
            if Y_test[i] == predictions[i]:
                correct += 1

        accuracy = correct / len(Y_test)
        print(accuracy)

        accuracies.append(accuracy)

    return sum(accuracies) / len(accuracies)

def gaussian_likelihood(X, mean, std):
    
    # ask about variants of this formula (i.e. whether std should be squared in first part, under sqrt, etc.)
    return (1 / (np.sqrt(2 * np.pi * std ** 2))) * np.exp(-1 * (X - mean) ** 2 / (2 * std ** 2))
    

def class_probabilities(means, std_devs, input_vector):

    probabilites = {}

    for class_index in range(len(means)):
        for i in range(len(means[0])):
            mean = means[class_index][i]
            std_dev = std_devs[class_index][i]

            x = input_vector[i]

            probabilites[class_index] = gaussian_likelihood(x, mean, std_dev)

    return probabilites

def predict(means, std_devs, input_vector):
    probs = class_probabilities(means, std_devs, input_vector)

    best_label, max_prob = None, -1

    for class_index, prob in probs.items():
        if best_label is None or prob > max_prob:
            best_label = class_index
            max_prob = prob

    return best_label

accuracy = naive_bayes(dataset, num_classes = 3)

print('Naive Bayes Average accuracy: ', accuracy)