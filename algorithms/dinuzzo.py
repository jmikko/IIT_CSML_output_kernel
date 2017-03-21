# This file is an implementation of the algorithm by Dinuzzo et al.
# contained in the paper "Learning output Kernels with Block Coordinate Descent"

import numpy as np
# import scipy as sp


class DinuzzoOutputKernel:
    """ Implementation of the algorithm by Dinuzzo et al.  contained in the paper
    Learning output Kernels with Block Coordinate Descent """

    def __init__(self, lam, delta, verbose=0):
        self.lam = lam
        self.delta = delta
        self.L = None
        self.C = None
        self.verbose = verbose

    def get_L(self):
        return self.L

    def solve_C_system(self, K, L, Y):
        """
        :param K:
        :param L:
        :param Y:
        :param lam:
        :return: the matrix C, solving Eq. (6)
        """
        LxK = np.kron(L.T, K)  # A composite array made of blocks of the second array scaled by the first.
        vY = np.matrix(np.reshape(Y, [-1, 1], order='F'))  # vectorized by rows (correct!)
        # vC = np.linalg.inv(LxK + self.lam * np.eye(LxK.shape[0])) * vY
        vC = np.linalg.solve(LxK + self.lam * np.eye(LxK.shape[0]), vY)  # We solve directly the linear system (stability)
        C = np.reshape(vC, Y.shape, order='F')  # by rows (correct!)
        return C

    def solve_Q_system(self, E, P):
        """
        :param E:
        :param P:
        :param lam:
        :return: the matrix Q, solving Eq. (8)
        """
        ETE = E.T * E
        ETE_one_dim = ETE.shape[0]
        # Q = np.linalg.pinv(ETE + self.lam * np.eye(ETE_one_dim)) * P
        Q = np.linalg.solve(ETE + self.lam * np.eye(ETE_one_dim), P)  # We solve directly the linear system (stability)
        return Q

    def run(self, Y, K):
        """
        :param Y:
        :param K:
        :param lam: trade-off between loss and regularization, see Eq. (1) (smaller means more importance for the loss)
        :param delta: stop condition (smaller means a more precise solution)
        :param verbose:
        :return:
        """
        L, C, Z = np.zeros((Y.shape[1], Y.shape[1])), np.zeros(Y.shape), np.zeros(Y.shape)
        err_matrix = Z + self.lam * C - Y
        step = 0
        max_iterations = 10000
        while np.linalg.norm(err_matrix, ord='fro') >= delta and step < max_iterations:
            C = self.solve_C_system(K, L, Y)
            if self.verbose > 1:
                diff = np.linalg.norm(Y - lam * C - K * C * L, ord='fro')
                print('Solution of C system with error: ', diff)
            E = K * C
            P = 0.5 * E.T * C - L
            Q = self.solve_Q_system(E, P)
            if self.verbose > 1:
                diff = np.linalg.norm((E.T * E + self.lam * np.eye(E.shape[1])) * Q - P, ord='fro')
                print('Solution of Q system with error: ', diff)
            L += self.lam * Q
            Z = E * L
            err_matrix = Z + self.lam * C - Y
            step += 1
            if self.verbose > 0:
                print('Obj. error step %d: %.8f' % (step, np.linalg.norm(err_matrix, ord='fro')))
        self.L = L
        self.C = C

    def prediction_function(self, Ktest):
        """
        :param L:
        :param C:
        :param Ktest: a Gram matrix with shape (training_examples, test_examples)
        :return:
        """
        return self.L * self.C.T * Ktest

    def classify(self, Ktest):
        return np.argmax(self.prediction_function(Ktest), axis=0)

if __name__ == "__main__":
    # Example of usage
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import OneHotEncoder
    iris_data = load_iris()
    enc = OneHotEncoder()

    X = np.matrix(iris_data.data)
    y = iris_data.target
    y = y.reshape(-1, 1)
    enc.fit(y)
    Y = enc.transform(y).toarray()

    K = X * X.T
    lam = 1.0
    delta = 1e-5

    dinuzzo = DinuzzoOutputKernel(lam=lam, delta=delta, verbose=2)
    dinuzzo.run(Y, K)
    print('Prediction: \n', dinuzzo.prediction_function(K[:, 0:3]))
    print('Classification: ', dinuzzo.classify(K[:, 0:3]))
    print('Matrix of correlation among tasks L: \n', dinuzzo.get_L())



