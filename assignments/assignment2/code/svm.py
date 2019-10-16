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

import os

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
            # exclude first column since it is not part of the feature vector
            X[fold].append(list(map(float, row[1:-1])))
            Y[fold].append(int(row[-1]))

    # evenly split dataset into k partitions
    avg, mod = divmod(len(dataset), k)

    folds = list(dataset[i * avg + min(i, mod):(i + 1) * avg + min(i + 1, mod)] for i in range(k))

    return folds

import seaborn as sns
import pandas as pd

# this function is based on https://stackoverflow.com/a/55766938/9464919, where 
# the user demonstrates creating plots of gridsearch parameters

# here, param_x appears on the x-axis, param_z appears on the lines (varied by color), and the mean_test_score (aka validation accuracy)
# is used as the dependent variable (on the y-axis)
def plot_cv_results(cv_results, param_x, param_z, metric='mean_test_score'):
    
    cv_results = pd.DataFrame(cv_results)
    col_x = 'param_' + param_x
    col_z = 'param_' + param_z
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    sns.pointplot(x=col_x, y=metric, hue=col_z, data=cv_results, ci=99, n_boot=64, ax=ax)
    ax.set_title("CV Grid Search Results")
    ax.set_xlabel(param_x)
    ax.set_ylabel(metric)
    ax.legend(title=param_z)
    
    return fig

# function to perform the actions of each step of the assignment (labeled below)
def kernel_testing(dataset):
    folds = generate_k_folds(dataset, k = 5)

    best_params = {
        'C': [], 
        'kernel': [], 
        'gamma': [], 
        'degree': []
    }

    results = {
        'ovo_acc': [], 
        'ovo_time': [], 
        'ovr_acc': [], 
        'ovr_time': [], 
        'balanced_ovo_acc': [], 
        'balanced_ovo_time': [], 
        'balanced_ovr_acc': [], 
        'balanced_ovr_time': []
    }

    # directory for all created figures
    os.makedirs('figures', exist_ok = True)

    # step 1: K-fold CV, grid search for hyperparameter optimization

    for i, fold in enumerate(folds):

        print('\nFOLD {}: '.format(i + 1))

        training_set = list(itertools.chain.from_iterable(folds[:i] +  folds[i + 1:]))
        test_set = fold

        X_train = [x[:-1] for x in training_set]
        Y_train = [y[-1] for y in training_set]
        X_test = [x[:-1] for x in test_set]
        Y_test = [y[-1] for y in test_set]

        params = {
            'kernel': ['rbf', 'poly', 'linear', 'sigmoid'], 
            'gamma': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], 
            'C': [0.01, 0.1, 1, 10, 100], 
            'degree': [2, 3, 4]
        }

        svc = svm.SVC()

        clf = GridSearchCV(svc, params, cv = 5)
        clf.fit(X_train, Y_train)

        print('Best params for curr fold: ', clf.best_params_)
        for key in clf.best_params_.keys():
            best_params[key].append(clf.best_params_[key])

        print('GridSearch optimal param validation accuracy: ', max(clf.cv_results_['mean_test_score']))
        print('\n\n')

        opt_svm = svm.SVC(
            C = clf.best_params_['C'], 
            kernel = clf.best_params_['kernel'], 
            gamma = clf.best_params_['gamma'], 
            degree = clf.best_params_['degree']        
        )


        fig = plot_cv_results(clf.cv_results_, 'gamma', 'C')
        plt.savefig('figures/fold{}_gamma_c.png'.format(i + 1))
        fig = plot_cv_results(clf.cv_results_, 'C', 'gamma')
        plt.savefig('figures/fold{}_c_gamma.png'.format(i + 1))


        # part 2: comparisons between OvO and OvR classifiers (acc. and training time)

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

        results['ovo_acc'].append(acc_ovo)
        results['ovr_acc'].append(acc_ovr)

        print('Training time (OVO): ', elapsed_ovo)
        print('Training time (OVR): ', elapsed_ovr)

        results['ovo_time'].append(elapsed_ovo)
        results['ovr_time'].append(elapsed_ovr)

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

        results['balanced_ovo_acc'].append(acc_balanced)
        results['balanced_ovo_time'].append(elapsed_balanced)

        opt_svm_balanced_ovr = svm.SVC(
            C = clf.best_params_['C'], 
            kernel = clf.best_params_['kernel'], 
            gamma = clf.best_params_['gamma'], 
            degree = clf.best_params_['degree'], 
            class_weight = 'balanced'
        )

        balanced_ovr_svm = OneVsRestClassifier(opt_svm_balanced_ovr)
        
        start = time.time()
        balanced_ovr_svm.fit(X_train, Y_train)
        elapsed_balanced_ovr = time.time() - start

        preds = balanced_ovr_svm.predict(X_test)

        correct = 0

        for i in range(len(preds)):
            if preds[i] == Y_test[i]:
                correct += 1

        acc_balanced_ovr = correct / len(preds)

        print('Accuracy (balanced OvR): ', acc_balanced_ovr)
        print('Training time (balanced OvR): ', elapsed_balanced_ovr)

        results['balanced_ovr_acc'].append(acc_balanced_ovr)
        results['balanced_ovr_time'].append(elapsed_balanced_ovr)


    # compute averages across each fold, per step
    print('\n\nAverage performances across 5-folds:\n')

    for key, item in results.items():
        print('Avg. {}: {}'.format(key, np.mean(item)))


    optimal = {}

    # not part of the assignment, but was interesting to find out
    # select the most used parameters for each category (popular vote)
    for key in best_params.keys():
        val, count = Counter(best_params[key]).most_common()[0]
        optimal[key] = val

    
    print('\n\nOPTIMAL PARAMETERS: \n', optimal)
    


kernel_testing(dataset)