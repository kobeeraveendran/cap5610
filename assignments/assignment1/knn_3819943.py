import numpy as np
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-k', type = int, nargs = 1, action = 'store', dest = 'k')
parser.add_argument('-m', type = str, nargs = 1, action = 'store', dest = 'mode')

dist_metric = 'Euclidean'

def euclidean_distance(p1, p2):

    e_dist = np.sqrt(np.sum((p1 - p2) ** 2))

    return e_dist


def cosine_distance(p1, p2):
    
    # check this later
    cos_similarity = (np.sum(p1 * p2)) / (np.sqrt(np.sum(p1 ** 2)) * np.sqrt(np.sum(p2 ** 2)))

    cos_dist = 1 - cos_similarity

    return cos_dist