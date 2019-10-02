import csv
from sklearn import svm
import random

def load_dataset():
    with open('../Data/glass.data') as f:
        lines = csv.reader(f)

        dataset = list(lines)

    '''
    X = []
    Y = []
    for row in dataset:
        X.append(list(map(float, row[1:-1])))
        Y.append(int(row[-1]))
    '''

    return dataset

dataset = load_dataset()
random.shuffle(dataset)

def generate_k_folds(dataset, k = 5):
    random.shuffle(dataset)
    
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

train_X, train_Y = dataset[: int(0.8 * len(dataset))]
test_X, test_Y = dataset[int(0.8 * len(dataset)):]

print('Training set: \n', train)
print('\n\nTest set: \n', test)