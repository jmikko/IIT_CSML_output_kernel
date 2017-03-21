import numpy as np
from functools import reduce
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
import os
from pkg1.utils import as_list
import _pickle as cpickle

PATH = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(PATH, 'Data')
print('data folder is', DATA_FOLDER)
CIFAR10_DIR = os.path.join(DATA_FOLDER, "CIFAR-10")


def _dim(what):
    sh = what.shape
    return sh[0] if len(sh) == 1 else sh


class Dataset:

    def __init__(self, data, target, sample_info_dicts=None, general_info_dict=None):
        """

        :param data: Numpy array containing data
        :param target: Numpy array containing targets
        :param general_info_dict: (optional) dictionary with further info about the dataset
        """
        self.data = data
        self.target = target
        self.sample_info_dicts = np.array([{}]*self.num_examples) if sample_info_dicts is None else sample_info_dicts
        assert len(self.data) == len(self.sample_info_dicts)
        assert len(self.data) == len(self.target)

        self.general_info_dict = general_info_dict or {}

    @property
    def num_examples(self):
        """

        :return: Number of examples in this dataset
        """
        return len(self.data)

    @property
    def dim_data(self):
        """

        :return: The data dimensionality as an integer, if input are vectors, or a tuple in the general case
        """
        return _dim(self.data[0])

    @property
    def dim_target(self):
        """

        :return: The target dimensionality as an integer, if targets are vectors, or a tuple in the general case
        """
        return _dim(self.target[0])


def to_one_hot_enc(seq):
    da_max = np.max(seq) + 1

    def create_and_set(_p):
        _tmp = np.zeros(da_max)
        _tmp[_p] = 1
        return _tmp
    return np.array([create_and_set(_v) for _v in seq])

def redivide_data(datasets, partition_proportions=None, shuffle=False, filters=None, maps=None):
    """
    Function that redivides datasets. Can be use also to shuffle or filter or map examples.

    :param datasets: original datasets, instances of class Dataset (works with get_data and get_targets for
    compatibility with mnist datasets
    :param partition_proportions: (optional, default None)  list of fractions that can either sum up to 1 or less
    then one, in which case one additional partition is created with proportion 1 - sum(partition proportions).
    If None it will retain the same proportion of samples found in datasets
    :param shuffle: (optional, default False) if True shuffles the examples
    :param filters: (optional, default None) filter or list of filters: functions with signature
    (data, target, index) -> boolean (accept or reject the sample)
    :param maps: (optional, default None) map or list of maps: functions with signature
    (data, target, index) ->  (new_data, new_target) (maps the old sample to a new one, possibly also to more
    than one sample, for data augmentation)
    :return: a list of datasets of length equal to the (possibly augmented) partition_proportion
    """

    all_data = np.vstack([get_data(d) for d in datasets])
    all_labels = np.vstack([get_targets(d) for d in datasets])

    all_infos = np.concatenate([d.sample_info_dicts for d in datasets]) #TODO: why?

    N = len(all_data)

    if partition_proportions:  # argument check
        sum_proportions = sum(partition_proportions)
        assert sum_proportions <= 1, "partition proportions must sum up to at most one: %d" % sum_proportions
        if sum_proportions < 1.: partition_proportions += [1. - sum_proportions]
    else:
        partition_proportions = [1.*len(get_data(d))/N for d in datasets]

    if shuffle:
        permutation = list(range(N))
        np.random.shuffle(permutation)

        all_data = np.array(all_data[permutation])
        all_labels = np.array(all_labels[permutation])
        all_infos = np.array(all_infos[permutation])

    if filters:
        filters = as_list(filters)
        data_triple = [(x, y, d) for x, y, d in zip(all_data, all_labels, all_infos)]
        for fiat in filters:
            data_triple = [xy for i, xy in enumerate(data_triple) if fiat(xy[0], xy[1], xy[2], i)]
        all_data = np.vstack([e[0] for e in data_triple])
        all_labels = np.vstack([e[1] for e in data_triple])
        all_infos = np.vstack([e[2] for e in data_triple])

    N = len(all_data)  # can be less now

    if maps:
        maps = as_list(maps)
        data_triple = [(x, y, d) for x, y, d in zip(all_data, all_labels, all_infos)]
        for _map in maps:
            data_triple = [_map(xy[0], xy[1], xy[2], i) for i, xy in enumerate(data_triple)]
        all_data = np.vstack([e[0] for e in data_triple])
        all_labels = np.vstack([e[1] for e in data_triple])
        all_infos = np.vstack([e[2] for e in data_triple])

    N = len(all_data)
    assert N == len(all_labels)

    calculated_partitions = reduce(
        lambda v1, v2: v1 + [sum(v1) + v2],
        [int(N*prp) for prp in partition_proportions],
        [0]
    )
    calculated_partitions[-1] = N

    print(calculated_partitions)

    print(len(all_data))

    new_general_info_dict = {}
    for data in datasets:
        new_general_info_dict = {**new_general_info_dict, ** data.general_info_dict}

    new_datasets = [
        Dataset(data=all_data[d1:d2], target=all_labels[d1:d2], sample_info_dicts=all_infos[d1:d2],
                general_info_dict=new_general_info_dict)
        for d1, d2 in zip(calculated_partitions, calculated_partitions[1:])
        ]

    return new_datasets


def load_cifar10(one_hot=True, partitions=None, filters=None, maps=None):
    path = CIFAR10_DIR + "/cifar-10.pickle"
    with open(path, "rb") as input_file:
        X, target_name, files = cpickle.load(input_file)
    dict_name_ID = {}
    i = 0
    list_of_targets = sorted(list(set(target_name)))
    for k in list_of_targets:
        dict_name_ID[k] = i
        i = i + 1
    dict_ID_name = {v: k for k, v in dict_name_ID.items()}
    Y = []
    for name_y in target_name:
        Y.append(dict_name_ID[name_y])
    if one_hot:
        Y = to_one_hot_enc(Y)
    dataset = Dataset(data = X, target = Y, general_info_dict= {'dict_name_ID':dict_name_ID, 'dict_ID_name':dict_ID_name},
                      sample_info_dicts=[{'target_name': t, 'files': f} for t, f in zip(target_name, files)])
    if partitions:
        res = redivide_data([dataset], partitions, filters=filters, maps=maps, shuffle=True)
        res += [None] * (3 - len(res))
        return Datasets(train=res[0], validation=res[1], test=res[2])
    return dataset


def get_data(d_set):
    if hasattr(d_set, 'images'):
        return d_set.images
    elif hasattr(d_set, 'data'):
        return d_set.data
    else:
        raise ValueError("something wrong with the dataset %s" % d_set)


def get_targets(d_set):
    if hasattr(d_set, 'labels'):
        return d_set.labels
    elif hasattr(d_set, 'target'):
        return d_set.target
    else:
        raise ValueError("something wrong with the dataset %s" % d_set)


def pad(_example, _size): return np.concatenate([_example] * _size)


