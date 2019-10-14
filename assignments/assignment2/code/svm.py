import csv
from sklearn import svm
import random
import itertools

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

def svm_clf(train, test, kernel, degree = 3, gamma = 'auto', class_weights = None, dec_fn_shape = 'ovo'):
    
    if kernel == 'poly':
        print('\n\nUsing kernel POLY with gamma = {} and degree = {}'.format(gamma, degree))

    else:
        print('\n\nUsing kernel {} with gamma = {}'.format(kernel.upper(), gamma))
    
    X_train, Y_train = train
    X_test, Y_test = test

    clf = svm.SVC(kernel = kernel, degree = degree, class_weight = class_weights, decision_function_shape = dec_fn_shape)
    clf.fit(X_train, Y_train)

    preds = clf.predict(X_test)

    #print('Predictions: \n', preds)
    #print('\n\nY_test: \n', Y_test)

    correct = 0

    for i in range(len(Y_test)):
        if Y_test[i] == preds[i]:
            correct += 1

    print('Accuracy: ', correct / len(Y_test))


def kernel_testing(dataset):
    folds = generate_k_folds(dataset, k = 5)

    for i, fold in enumerate(folds):

        training_set = list(itertools.chain.from_iterable(folds[:i] +  folds[i + 1:]))
        test_set = fold

        X_train = [x[:-1] for x in training_set]
        Y_train = [y[-1] for y in training_set]

        X_test = [x[:-1] for x in test_set]
        Y_test = [y[-1] for y in test_set]

        train = (X_train, Y_train)
        test = (X_test, Y_test)

        '''
        params = { 
            'degree': [3, 4, 5], 
            'gamma': ['auto', 'scale']
        }
        '''

        for kernel in ['rbf', 'linear', 'poly', 'sigmoid']:
            
            if kernel == 'poly':
                for degree in [3, 4, 5]:
                    svm_clf(train, test, kernel = kernel, degree = degree)

            for gamma in ['auto', 'scale']:
                svm_clf(train, test, kernel = kernel, gamma = gamma)


kernel_testing(dataset)