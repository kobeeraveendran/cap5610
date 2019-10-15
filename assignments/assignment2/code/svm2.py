import csv
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from collections import Counter
import time

# in case FutureWarning messages show for deprecations of 'gamma'
from warnings import simplefilter
simplefilter(action = 'ignore', category = FutureWarning)
simplefilter(action = 'ignore', category = DeprecationWarning)

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

    best_params = {
        'C': [], 
        'kernel': [], 
        'gamma': [], 
        'degree': []
    }

    best_acc = 0
    best_params_acc = {}

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
            'degree': [2, 3, 4]
        }

        svc = svm.SVC()

        clf = GridSearchCV(svc, params, cv = 5)
        clf.fit(X_train, Y_train)

        print(clf.best_params_)
        for key in clf.best_params_.keys():
            best_params[key].append(clf.best_params_[key])

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

        # heatmap of c vs. gamma

        print('MEAN SCORES SHAPE: ', mean_scores.shape)

        scores = mean_scores.reshape(len(params['C']), len(params['gamma']))
        plt.figure(fisize = (8, 6))
        plt.subplots_adjust(left = .2, right = 0.95, bottom = 0.15, top = 0.95)
        plt.imshow(scores, interpolation = 'nearest', cmap = plt.cm.hot)
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(params['gamma'])), params['gamma'], rotation = 45)
        plt.yticks(np.arange(len(params['C'])), params['C'])
        plt.show()

        '''
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        plt.show()
        '''

        # extract params for best-performing kernel (for plot)
        #for i in range(len(mean_scores)):
        #    print()

        # part 2 comparisons

        ovo_svm = OneVsOneClassifier(opt_svm)

        start = time.time()
        ovo_svm.fit(X_train, Y_train)
        elapsed_ovo = time.time() - start

        preds = ovo_svm.predict(X_test)

        correct = 0

        for i in range(len(preds)):
            if preds[i] == Y_test[i]:
                correct +=  1

        acc_ovo = correct / len(preds)

        opt_svm = svm.SVC(
            C = clf.best_params_['C'], 
            kernel = clf.best_params_['kernel'], 
            gamma = clf.best_params_['gamma'], 
            degree = clf.best_params_['degree']
        )

        ovr_svm = OneVsRestClassifier(opt_svm)

        start = time.time()
        ovr_svm.fit(X_train, Y_train)
        elapsed_ovr = time.time() - start

        preds = ovr_svm.predict(X_test)

        correct = 0

        for i in range(len(preds)):
            if preds[i] == Y_test[i]:
                correct += 1

        acc_ovr = correct / len(preds)

        print('Accuracy (OVO): ', acc_ovo)
        print('Accuracy (OVR): ', acc_ovr)

        print('Training time (OVO): ', elapsed_ovo)
        print('Training time (OVR): ', elapsed_ovr)

        # part 4
        opt_svm_balanced = svm.SVC(
            C = clf.best_params_['C'], 
            kernel = clf.best_params_['kernel'], 
            gamma = clf.best_params_['gamma'], 
            degree = clf.best_params_['degree'], 
            class_weight = 'balanced'
        )

        start = time.time()
        opt_svm_balanced.fit(X_train, Y_train)
        elapsed_balanced = time.time() - start

        preds = opt_svm_balanced.predict(X_test)

        correct = 0

        for i in range(len(preds)):
            if preds[i] == Y_test[i]:
                correct += 1

        acc_balanced = correct / len(preds)

        print('Accuracy (balanced): ', acc_balanced)
        print('Training time (balanced): ', elapsed_balanced)

        #if acc > best_acc:
        #    best_acc = acc
        #    best_params_acc = clf.best_params_



    optimal = {}

    # select the most popular parameters for each parameter category
    for key in best_params.keys():
        val, count = Counter(best_params[key]).most_common()[0]
        optimal[key] = val

    # OR select params that yielded the highest accuracy (above)

    print('\n\nOPTIMAL PARAMETERS: \n', optimal)

    print('\n\nADDITIONAL OPTIMAL PARAMETER (HIGHEST ACC): \n', best_params_acc)

    return optimal, best_params_acc

kernel_testing(dataset)