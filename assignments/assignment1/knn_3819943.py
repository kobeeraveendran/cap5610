import numpy as np
import math
import argparse

from dataset_preprocessing import load_dataset, train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-k', type = int, nargs = 1, action = 'store', dest = 'k')
parser.add_argument('-m', type = str, nargs = 1, action = 'store', dest = 'mode')

dist_metric = 'Euclidean'

def euclidean_distance(p1, p2):

    return np.sqrt(np.sum((p1 - p2) ** 2))


def cosine_distance(p1, p2):
    
    return 1 - (np.dot(p1, p2) / (np.sqrt(np.dot(p1, p1)) * np.sqrt(np.dot(p2, p2))))

