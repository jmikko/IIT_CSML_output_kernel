import matplotlib.pyplot as plt
from IPython.display import IFrame
import IPython
import gzip
import os
import _pickle as pickle
import numpy as np


def save_obj(obj, name, default_overwrite=False):
    filename = '%s.pkgz' % name
    if not default_overwrite and os.path.isfile(filename):
        overwrite = input('A file named %s already exists. Overwrite (Leave string empty for NO!)?' % filename)
        if not overwrite:
            print('No changes done.')
            return
        print('Overwriting...')
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)
        print('File saved!')


def load_obj(name):
    filename = '%s.pkgz' % name
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def save_adjacency_matrix_for_gephi(matrix, name, class_names=None):
    filename = '%s.csv' % name
    m, n = np.shape(matrix)
    assert m == n, '%s should be a square matrix.' % matrix
    if not class_names:
        class_names = [str(k) for k in range(n)]
    left = np.array([class_names]).T
    matrix = np.hstack([left, matrix])
    up = np.vstack([[''], left]).T
    matrix = np.vstack([up, matrix])
    np.savetxt(filename, matrix, delimiter=';', fmt='%s')