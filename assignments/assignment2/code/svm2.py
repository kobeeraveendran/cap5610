import csv
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import random
import itertools
import time

# in case FutureWarning messages show for deprecations of 'gamma'
from warnings import simplefilter
simplefilter(action = 'ignore', category = FutureWarning)

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
    results = {}

    for i, fold in enumerate(folds):

        training_set = list(itertools.chain.from_iterable(folds[:i] +  folds[i + 1:]))
        test_set = fold

        X_train = [x[:-1] for x in training_set]
        Y_train = [y[-1] for y in training_set]
        X_test = [x[:-1] for x in test_set]
        Y_test = [y[-1] for y in test_set]

        #print('X_train: \n', X_train)

        #print('\n\nY_train: \n', Y_train)

        train = (X_train, Y_train)
        test = (X_test, Y_test)

        params = {
            'kernel': ['rbf', 'poly', 'linear', 'sigmoid'], 
            'gamma': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], 
            'C': [0.01, 0.1, 1, 10, 100], 
            'degree': [3, 4]
        }

        svc = svm.SVC()

        clf = GridSearchCV(svc, params, cv = 5)
        clf.fit(X_train, Y_train)

        print(clf.best_params_)
        print(max(clf.cv_results_['mean_test_score']))

        mean_scores = clf.cv_results_['mean_test_score']
        std_scores = clf.cv_results_['std_test_score']
        tested_params = clf.cv_results_['params']

        #for i in range(len(mean_scores)):
        #    print('{:.3f} (+/-){:.3f} for {}'.format(mean_scores[i], std_scores[i] * 2, tested_params[i]))

        opt_svm = svm.SVC(
            C = clf.best_params_['C'], 
            kernel = clf.best_params_['kernel'], 
            gamma = clf.best_params_['gamma'], 
            degree = clf.best_params_['degree']        
        )
        opt_svm.fit(X_train, Y_train)

        preds = opt_svm.predict(X_test)

        correct = 0

        for i in range(len(preds)):
            if preds[i] == Y_test[i]:
                correct +=  1

        acc = correct / len(preds)

        print('Accuracy: ', acc)

    '''
    print('\n\nOne vs. One results: \n')
    for key, item in results.items():
        #print(key + ': ', item)
        print(key + ' avg. acc.: ', np.mean(item[0][::2]))
        print(key + ' avg. training time: ', np.mean(item[1][::2]))


    print('\n\nOne vs. Rest results: \n')
    for key, item in results.items():
        print(key + ' OvR accuracy (avg): ', np.mean(item[0][1::2]))
        print(key + ' OvR training time (avg): ', np.mean(item[1][1::2]))

    '''


kernel_testing(dataset)