import csv
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import random
import itertools
import time

def load_dataset():
    with open('../Data/glass.data') as f:
        lines = csv.reader(f)

        dataset = list(lines)

    # convert datatype of columns
    for i in range(len(dataset)):
        label = dataset[i][-1]
        dataset[i] = dataset[i][1:]
        dataset[i] = [float(x) for x in dataset[i][:-1]]
        dataset[i].append(int(label))

    return dataset

dataset = load_dataset()

def generate_k_folds(dataset, k = 5):
    random.shuffle(dataset)

    X = [[] for i in range(k)]
    Y = [[] for i in range(k)]

    for fold in range(k):
        for row in dataset:
            X[fold].append(list(map(float, row[1:-1])))
            Y[fold].append(int(row[-1]))

    avg, mod = divmod(len(dataset), k)

    folds = list(dataset[i * avg + min(i, mod):(i + 1) * avg + min(i + 1, mod)] for i in range(k))

    return folds

def train_svm(train, test, kernel_type, class_weights = None, decision_fn_shape = 'ovo'):

    train_X, train_Y = train
    test_X, test_Y = test

    if class_weights:
        clf = svm.SVC(
            kernel = kernel_type, 
            class_weight = class_weights, 
            decision_function_shape = decision_fn_shape
            )

    else:
        clf =  svm.SVC(
            kernel = kernel_type, 
            decisidecision_function_shape = decision_fn_shape
            )

    clf.fit(train_X, train_Y)

    predictions = []
    print('Predictions')

    for sample in test_X:
        pred = clf.predict(sample)
        predictions.append(pred)
        print(pred)

    params = clf.get_params()

def svm_clf(train, test, kernel, c = None, degree = 3, gamma = 'auto', class_weights = None, ovr = False):
    
    if kernel == 'poly':
        print('\n\nUsing kernel POLY with gamma = {} and degree = {}'.format(gamma, degree))

    else:
        print('\n\nUsing kernel {} with gamma = {}'.format(kernel.upper(), gamma))
    
    X_train, Y_train = train
    X_test, Y_test = test

    start = time.time()

    if not ovr:        
        clf = svm.SVC(kernel = kernel, degree = degree, class_weight = class_weights)

    else:
        clf = OneVsRestClassifier(svm.SVC(kernel = kernel, degree = degree, class_weight = class_weights))
    clf.fit(X_train, Y_train)

    time_elapsed = time.time() - start

    preds = clf.predict(X_test)

    #print('Predictions: \n', preds)
    #print('\n\nY_test: \n', Y_test)

    correct = 0

    for i in range(len(Y_test)):
        if Y_test[i] == preds[i]:
            correct += 1

    acc = correct / len(Y_test)

    #print('Accuracy: ', acc)

    return acc * 100, time_elapsed

# function to perform all the actions of step 1 of the assignment
def kernel_testing(dataset):
    folds = generate_k_folds(dataset, k = 5)

    # results for each hyperparameter are in the form ([accuracies], [training times])
    results = {
        'rbf_auto': ([], []),
        'rbf_scale': ([], []), 
        'linear_auto': ([], []), 
        'linear_scale': ([], []), 
        'poly_degree3': ([], []), 
        'poly_degree4': ([], []), 
        'poly_degree5': ([], []), 
        'poly_auto': ([], []), 
        'poly_scale': ([], []), 
        'sigmoid_auto': ([], []), 
        'sigmoid_scale': ([], [])
    }

    for i, fold in enumerate(folds):

        training_set = list(itertools.chain.from_iterable(folds[:i] +  folds[i + 1:]))
        test_set = fold

        X_train = [x[:-1] for x in training_set]
        Y_train = [y[-1] for y in training_set]
        X_test = [x[:-1] for x in test_set]
        Y_test = [y[-1] for y in test_set]

        train = (X_train, Y_train)
        test = (X_test, Y_test)

        for kernel in ['rbf', 'linear', 'poly', 'sigmoid']:
            
            if kernel == 'poly':
                for degree in [3, 4, 5]:
                    key = 'poly_degree' + str(degree)
                    # part 1 SVMs
                    acc, time_elapsed = svm_clf(train, test, kernel = kernel, degree = degree)
                    results[key][0].append(acc)
                    results[key][1].append(time_elapsed)

                    # part 2 SVMs (same as part1 but with one vs. rest)
                    acc, time_elapsed = svm_clf(train, test, kernel = kernel, degree = degree, ovr = True)
                    results[key][0].append(acc)
                    results[key][1].append(time_elapsed)


            for gamma in [0.01, 0.1, 1, 10, 100]:
                key = kernel + '_' + gamma
                # part 1 SVMs
                acc, time_elapsed = svm_clf(train, test, kernel = kernel, gamma = gamma)
                results[key][0].append(acc)
                results[key][1].append(time_elapsed)

                acc, time_elapsed = svm_clf(train, test, kernel = kernel, ovr = True)
                results[key][0].append(acc)
                results[key][1].append(time_elapsed)

            for c in [0.01, 0.1, 1, 10, 100]:
                key = kernel + '_c_' + str(c)

                acc, time_elapsed = svm_clf(train, test, kernel = kernel, c = c)
                results[key][0].append(acc)
                results[key][1].append(time_elapsed)

                acc, time_elapsed = svm_clf(train, test, kernel = kernel, c = c, ovr = True)
                results[key][0].append(acc)
                results[key][1].append(time_elapsed)

    print('\n\nOne vs. One results: \n')
    for key, item in results.items():
        #print(key + ': ', item)
        print(key + ' avg. accuracy over 5 folds: ', np.mean(item[0][::2]))
        print(key + ' avg. training time over 5 folds: ', np.mean(item[1][::2]))


    print('\n\nOne vs. Rest results: \n')
    for key, item in results.items():
        print(key + ' one-vs-rest accuracy (avg): ', np.mean(item[0][1::2]))
        print(key + ' one-vs-rest training time (avg): ', np.mean(item[1][1::2]))



kernel_testing(dataset)