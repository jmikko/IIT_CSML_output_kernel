import scipy.io
from save_and_load import load_obj

data = load_obj('/home/michele/PycharmProjects/IIT_CSML_output_kernel/Data/CIFAR-10/partitions/datasets')
scipy.io.savemat('./datasets_10.mat', mdict={'all': data})
