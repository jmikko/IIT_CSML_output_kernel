import collections
import sys
from enum import Enum
from functools import reduce
from time import gmtime, strftime
import numpy as np
import tensorflow as tf



# TODO add another setting table to be stored into a file, with e.g. the path for saving file
# (e.g. /media/luca/DATA/____)
settings = {
    'WSA': True,  # gives a warning when new nodes are being created during session runtime
    'NOTEBOOK_TITLE': ''
}


def wsr(node):  # warning on session running
    if settings['WSA'] and tf.get_default_session():
        print('Warning: creating nodes at tf.Session runtime: node %s' % node,
              file=sys.stderr)
    return node


def simple_name(tensor):
    return tensor.name.split(':')[0]


class SummaryUtil:
    def __init__(self, ops=None, condition=None, writer=None, fd_supplier=None):
        """
        Utility class used by SummariesUtils to collect summary ops

        :param ops: summary operations
        :param condition: (default: always True)
        :param writer: either a tf.summary.FileWriter or a string for standard log dirs
        :param fd_supplier: (default None supplier)
        """
        assert ops is not None
        self.ops = ops
        self.condition = condition if condition else lambda step: True
        self.writer = (tf.summary.FileWriter(writer + strftime("%a_%d_%b_%Y_%H:%M:%S", gmtime()))
                       if isinstance(writer, str) else writer)
        self.fd_supplier = fd_supplier if fd_supplier else lambda: None


class SummaryUtils:

    def __init__(self, *summary_utils_list):
        self.summary_list = summary_utils_list

    def run(self, session, step):
        [su.writer.add_summary(
            session.run(su.ops, feed_dict=su.fd_supplier(step)), step
        ) for su in self.summary_list if su.condition(step)]


PrintUtil = collections.namedtuple('PrintSummaryUtil', ['print_supplier', 'condition'])


def stepwise_pu(print_supplier, every_n_steps):
    return PrintUtil(print_supplier=print_supplier,
                     condition=lambda step: step % every_n_steps == 0)


def unconditional_pu(print_supplier):
    return PrintUtil(print_supplier=print_supplier, condition=lambda step: True)


class PrintUtils:  # TODO fix this class... Looks horrible now

    def __init__(self, *print_list, add_print=False):  # here can be added also the standard output
        self.print_utils = print_list
        self.add_print = add_print

    def run(self, session=None, step=None):
        if self.add_print:
            [print(pu.print_supplier(session, step)) for pu in self.print_utils if pu.condition(step)]
        else:
            [pu.print_supplier(session, step) for pu in self.print_utils if pu.condition(step)]


class MergedUtils:

    def __init__(self, *utils):
        self.merged = utils

    def run(self, session, step):
        [u.run(session, step) for u in self.merged]


def norm(v, name='norm'):
    """
    The the norm of a Tensor: if v is a vector then the norm is the Euclid's norm L2, otherwise it computes the
    Frobenius norm.

    :param name: (optional, default norm) name of the name_scope
    :param v: tf.Tensor or Variable
    :return: a tensor that computes the norm
    """
    with tf.name_scope(name):
        return wsr(tf.sqrt(tf.reduce_sum(tf.square(v))))


def cross_entropy_loss(y, targets, linear_input=True, eps=1.e-5, name='cross_entropy_loss'):
    """
    Clipped standard-version cross entropy loss. Implemented because  the standard function
    tf.nn.softmax_cross_entropy_with_logits has wrong (?) Hessian.
    Clipped because it easily brings to nan otherwise, especially when calculating the Hessian.

    Maybe the code could be optimized since ln(softmax(z_j)) = z_j - prod z_i . Should benchmark it.

    :param y: softmax or linear output of the model
    :param targets: labels
    :param linear_input: True is y is linear in which case tf.nn.softmax will be applied to y
    :param eps: (optional, default 1.e-5) clipping value for log.
    :param name: (optional, default cross_entropy_loss) name scope for the defined operations.
    :return: tensor for the cross_entropy_loss (WITHOUT MEAN ON THE EXAMPLES)
    """
    with tf.name_scope(name):
        softmax_out = tf.nn.softmax(y) if linear_input else y
        return -tf.reduce_sum(
            targets * tf.log(tf.clip_by_value(softmax_out, eps, 1. - eps)), reduction_indices=[1]
        )


def binary_cross_entropy(y, targets, linear_input=True, eps=1.e-5, name='binary_cross_entropy_loss'):
    """
    Same as cross_entropy_loss for the binary classification problem. the model should have a one dimensional output,
    the targets should be given in form of a matrix of dimensions batch_size x 1 with values in [0,1].

    :param y: sigmoid or linear output of the model
    :param targets: labels
    :param linear_input: (default: True) is y is linear in which case tf.nn.sigmoid will be applied to y
    :param eps: (optional, default 1.e-5) clipping value for log.
    :param name: (optional, default binary_cross_entropy_loss) name scope for the defined operations.
    :return: tensor for the cross_entropy_loss (WITHOUT MEAN ON THE EXAMPLES)
    """
    with tf.name_scope(name):
        sigmoid_out = tf.nn.sigmoid(y)[:, 0] if linear_input else y
        return - (targets[:, 0]*tf.log(tf.clip_by_value(sigmoid_out, eps, 1. - eps)) +
                  (1. - targets[:, 0])*tf.log(tf.clip_by_value(1. - sigmoid_out, eps, 1. - eps)))



def dot(v1, v2):
    v1_shape = v1.get_shape().ndims
    v2_shape = v2.get_shape().ndims
    # print(v1_shape, v2_shape)
    if v1_shape > 1 and v2_shape > 1:
        return tf.matmul(v1, v2)
    elif v1_shape == 2 and v2_shape == 1:
        return tf.reduce_sum(v1 * v2, reduction_indices=[1])
    elif v1_shape == 1 and v2_shape == 2:  # fine for mat
        return tf.reduce_sum(v1 * tf.transpose(v2), reduction_indices=[1])
    elif v1_shape == 1 and v2_shape == 1:
        return tf.reduce_sum(v1 * v2)
    else:
        raise NotImplementedError()  # TODO finish implement this also with scalars and maybe with others


def vectorize_all(var_list):
    """Given a list of tensors returns their concatenated vectorization.
    Note that for matrices the vectorization is row-wise instead of column-wise as
    it should be in Magnus. Could it be a problem?

    :param var_list: **bold**

    :return: vectorization of `var_list`"""
    return wsr(tf.concat(0, [tf.reshape(_w, [-1]) for _w in var_list]))


def hv_1(_dyn, _lv, _v):  # NOTE this is not an efficient implementation
    """Computes hessian-vector product (without storing the Hessian)
    in a naive way. If _lv is a list of tensor, then vectorizes them with vectorize_all"""
    res = []
    for i in range(_v.get_shape()[0].value):
        _hvi = tf.gradients(_dyn[i], _lv)
        if isinstance(_lv, list):
            _hvi = vectorize_all(_hvi)
        res.append(
            tf.reduce_sum(_hvi * _v)
        )
    return tf.pack(res)  # takes forever....


def canonical_base(n):
    identity = np.eye(n)
    return [tf.constant(identity[:, j], dtype=tf.float32) for j in range(n)]


# ChunksInfo = collections.namedtuple('ChunksInfo', ['start', 'end', 'reshape'])

def as_list(obj):
    return obj if isinstance(obj, list) else [obj]


def reshape_generator(original_var, start, end):
    return lambda merged: wsr(tf.reshape(merged[start:end], original_var.get_shape()))


def var_or_merged(v):
    return v.tensor if isinstance(v, MergedVariable) else v

Vl_Mode = Enum('Vl_Mode', 'RAW BASE TENSOR')  # allowed modes for MergedVariable.var_list


class MergedVariable:

    def __init__(self, var_list, name='merged_variable'):
        self._var_list = var_list
        self.tensor = tf.identity(vectorize_all([var_or_merged(v) for v in var_list]), name=name)

        self.chunks_info_dict = {}

        start = 0

        for v in self.var_list(Vl_Mode.BASE):  # CHANGED (in var_list)
            dim_var = reduce(lambda v1, v2: v1 * v2, v.get_shape().as_list(), 1)
            end = start + dim_var

            self.chunks_info_dict[v] = reshape_generator(v, start, end)

            start += dim_var

    def var_list(self, mode=Vl_Mode.RAW):
        """
        Get the chunks that define this variable.

        :param mode: (optional, default VL_MODE.RAW) VL_MODE.RAW: returns simply var_list, that may contain tf.Variables
                         or MergedVariables
                     VL_MODE.BASE: returns a list of tf.Variables that are the "base" variables that for this
                     MergedVariable
                     VL_MODE.TENSOR: returns a list of tf.Variables or tf.Tensor from the MergedVariables
        :return: A list that may contain tf.Tensors, tf.Variables and/or MergedVariables
        """
        if mode == Vl_Mode.RAW:
            return self._var_list
        elif mode == Vl_Mode.BASE:
            return self._get_base_variable_list()
        elif mode == Vl_Mode.TENSOR:
            return self._var_list_as_tensors()
        else:
            raise NotImplementedError('mode %d does not exists' % mode)

    def _get_base_variable_list(self):
        """
        This methods checks that all the elements of var_list are legitimate (tf.Variables or MergedVariables)
        and returns the underlying tf.Variables.
        :return:
        """
        res = []
        for v in self._var_list:
            if isinstance(v, MergedVariable):
                res.extend(v._get_base_variable_list())
            elif isinstance(v, tf.Variable):
                res.append(v)
            else:
                raise ValueError('something wrong here')
        return res

    def _var_list_as_tensors(self):
        if any([isinstance(v, MergedVariable) for v in self._var_list]):
            return [var_or_merged(v) for v in self._var_list]
        else:
            return [self.tensor]

    def generate_swap_dict(self):
        return {v.op.outputs[0]: reshape(self.tensor) for v, reshape in self.chunks_info_dict.items()}

    def assign(self, value, use_locking=False):
        return [
            wsr(v.assign(reshape(value), use_locking=use_locking)) for v, reshape in self.chunks_info_dict.items()
            ]

    def eval(self, feed_dict=None):
        return self.tensor.eval(feed_dict=feed_dict)

    @property
    def graph(self):
        return self.tensor.graph

