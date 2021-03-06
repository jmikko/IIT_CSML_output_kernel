# This file is an implementation of the algorithm by Dinuzzo et al.
# contained in the paper "Learning output Kernels with Block Coordinate Descent"

import numpy as np
from scipy.linalg import solve_sylvester
# import scipy as sp


class DinuzzoOutputKernel:
    """ Implementation of the algorithm by Dinuzzo et al.  contained in the paper
    Learning output Kernels with Block Coordinate Descent """

    def __init__(self, lam, delta, max_iterations=10000, verbose=0):
        self.lam = lam
        self.delta = delta
        self.L = None
        self.C = None
        self.max_iterations = max_iterations
        self.verbose = verbose

    def get_L(self):
        return self.L

    def solve_C_system_old(self, K, L, Y):
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
        vC = np.linalg.solve(LxK + self.lam * np.eye(LxK.shape[0]), vY)  # We solve the linear system
        C = np.reshape(vC, Y.shape, order='F')  # by rows (correct!)
        return C

    def solve_C_system(self, Kinv, L, KinvY):
        """
        :param Kinv:
        :param L:
        :param KinvY:
        :param lam:
        :return: the matrix C, solving Eq. (6) in the form C * L + (K^-1 * lambda) * C = K^-1 * Y as a Sylvester eq.
        """
        C = np.matrix(solve_sylvester(Kinv * self.lam, L, KinvY))
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
        Kinv = np.linalg.pinv(K)
        KinvY = Kinv * Y
        while np.linalg.norm(err_matrix, ord='fro') >= self.delta and step < self.max_iterations:
            C = self.solve_C_system(Kinv, L, KinvY)
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
        return np.array(np.argmax(self.prediction_function(Ktest), axis=0))[0]

if __name__ == "__main__":
    # Example of usage
    from sklearn.metrics import accuracy_score

    # With CIFAR-10
    CIFAR10 = False
    if CIFAR10:
        from datasets import load_cifar10, Datasets, Dataset
        import matplotlib.pyplot as plt

        datasets = load_cifar10(one_hot=True, partitions=[0.01, 0.01])
        training_set_X = np.matrix(datasets[0].data)
        training_set_Y = datasets[0].target
        validation_set_X = np.matrix(datasets[1].data)
        validation_set_Y = datasets[1].target
        test_set_X = np.matrix(datasets[2].data)
        test_set_Y = datasets[2].target
        Ktr = training_set_X * training_set_X.T
        lam = 100.0
        delta = 1e-5
        Ktrte = training_set_X * test_set_X.T

        dinuzzo = DinuzzoOutputKernel(lam=lam, delta=delta, verbose=1)
        dinuzzo.run(training_set_Y, Ktr)
        print('Prediction: \n', dinuzzo.prediction_function(Ktrte))
        print('Classification: ', dinuzzo.classify(Ktrte))

        L = np.array(dinuzzo.get_L())
        print('Matrix of correlation among tasks L: \n', L)

        classify = dinuzzo.classify(Ktrte)
        y_true = np.argmax(test_set_Y, axis=1)
        accuracy_test = accuracy_score([y for y in y_true], [y for y in classify])

        classify = dinuzzo.classify(Ktr)
        y_true = np.argmax(training_set_Y, axis=1)
        accuracy_training = accuracy_score([y for y in y_true], [y for y in classify])

        print('Train accuracy: ', accuracy_training)
        print('Test accuracy: ', accuracy_test)

        dict_ID_name = datasets[2].general_info_dict['dict_ID_name']
        print('Correspondence ID_class - Class:\n', dict_ID_name)
        # Plots:
        np.fill_diagonal(L, 0.0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(L, interpolation='nearest', cmap=plt.cm.gray)
        fig.colorbar(cax)
        plt.title('Matrix of correlation among tasks L')
        alpha = ['' + dict_ID_name[2 * k] + ' ' + dict_ID_name[2 * k + 1] for k in range(5)]
        ax.set_xticklabels([''] + alpha)
        ax.set_yticklabels([''] + alpha)
        plt.show()

    # With iris
    IRIS = True
    if IRIS:
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import accuracy_score

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

        dinuzzo = DinuzzoOutputKernel(lam=lam, delta=delta, max_iterations=1000, verbose=1)
        dinuzzo.run(Y, K)
        print('Prediction: \n', dinuzzo.prediction_function(K[:, 0::10]))
        print('Classification: ', dinuzzo.classify(K[:, 0::10]))
        print('Matrix of correlation among tasks L: \n', dinuzzo.get_L())

        y_true = np.argmax(Y, axis=1)
        classify = dinuzzo.classify(K)
        accuracy_train = accuracy_score([y for y in y_true], [y for y in classify])
        print('Accuracy training set:',accuracy_train)



