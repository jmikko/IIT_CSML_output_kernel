# This file is an implementation of the algorithm by Jawanpuria et al.
# contained in the paper "Efficient Output Kernel Learning for Multiple Task"

import numpy as np
import scipy as sp
from scipy.optimize import minimize_scalar, newton
# import cvxopt

class EfficientOutputKernel:
    """ Implementation of the algorithm by Jawanpuria et al.
    contained in the paper "Efficient Output Kernel Learning for Multiple Task """

    def __init__(self, lam, C, k, max_iterations=1000, verbose=0):
        self.lam = lam
        self.C = C
        self.k = k
        self.beta = None
        self.omega = None
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.max_sub_iterations = 100

    def get_beta(self):
        return self.beta

    def get_omega(self):
        return self.omega

    def solve_delta(self, K, alpha, tasks, labels, idx_per_task, i):
        a = K[i, i]
        R = K.shape[0]
        J = K.shape[1]
        n_tasks = len(idx_per_task)
        # b = np.array([[np.sum([K[r, j] * alpha[j] for j in range(J) if tasks[j] == s])
        #               for r in range(R)] for s in range(n_tasks)])
        # b = b.T
        b = np.array([[np.sum([K[i, j] * alpha[j] for j in range(J) if tasks[j] == s])
                       for r in range(n_tasks)] for s in range(n_tasks)])
        b = b.T
        c = np.array([[np.matrix(alpha[idx_per_task[s]]) * np.matrix(K[idx_per_task[s], :][:, idx_per_task[z]])
                       * np.matrix(alpha[idx_per_task[z]]).T for s in range(n_tasks)] for z in range(n_tasks)])
        # valu = [[(np.matrix(alpha[idx_per_task[s]]), np.matrix(K[idx_per_task[s], :][:, idx_per_task[z]]),
        #          np.matrix(alpha[idx_per_task[z]]).T) for s in range(n_tasks)] for z in range(n_tasks)]
        # print(valu)
        c = c.reshape((c.shape[0], c.shape[1]))
        eta = (self.lam / (self.C * (4 * self.k - 2))) * ((2 * self.k - 1) / (2 * self.k * self.lam))**(2 * self.k)

        # print('b shape', b.shape)
        # print('C shape', c.shape)
        # print('a', a)
        # print('b', b)
        # print('c', c)
        # print('eta', eta)

        def L_conj(v):  # Conjugate function of the loss
            return (v**2.0 / 2.0) + (labels[i] * v)

        def F(delta):
            r = tasks[i]
            res = L_conj((-alpha[i] - delta) / self.C)
            res += float(eta * ((a * delta**2 + 2 * b[r, r] * delta + c[r, r])**(2 * self.k)))
            res += eta * (2 * np.sum([(b[r, s] * delta + c[r, s])**(2 * self.k)
                                      for r in range(n_tasks) for s in range(n_tasks) if s != r]))
            # res += eta * np.sum([c[s, z]**(2 * self.k) for s in range(n_tasks)
            #                     for z in range(n_tasks) if s != r and z != r])  # Useless for the "argmin" on delta
            return res

        def dL_conj(v):
            return v - (labels[i] / self.C)

        def dF(delta):
            r = tasks[i]
            res = dL_conj((alpha[i] + delta) / self.C**2)
            res += float(eta * (2 * self.k * (a * delta**2 + 2 * b[r, r] * delta + c[r, r])**(2 * self.k - 1))
                         * (2 * a * delta + 2 * b[r, r]))
            res += eta * (2 * 2 * self.k * np.sum([(b[r, s] * delta + c[r, s])**(2 * self.k - 1) * b[r, s]
                                              for r in range(n_tasks) for s in range(n_tasks) if s != r]))
            return res

        def ddF(delta):
            r = tasks[i]
            res = 1.0 / self.C**2
            res += float(eta * (2 * self.k * (2 * self.k - 1) * (a * delta**2 + 2 * b[r, r] * delta + c[r, r])**(2 * self.k - 2) * (2 * a * delta + 2 * b[r, r])**2 +
                                2 * self.k * (a * delta**2 + 2 * b[r, r] * delta + c[r, r])**(2 * self.k - 1) * 2 * a))
            res += eta * (2 * 2 * self.k * (2 * self.k - 1) * np.sum([(b[r, s] * delta + c[r, s])**(2 * self.k - 2) *
                                                                      b[r, s]**2 for r in range(n_tasks)
                                                                      for s in range(n_tasks) if s != r]))
            return res

        min_fun = False
        if min_fun:
            res = minimize_scalar(F)
            res = res['x']
        else:  # newton => zeros in the dF, given the ddF
            try:
                res = newton(dF, 0.0, fprime=ddF, maxiter=self.max_sub_iterations, tol=1e-6)  # minimize_scalar(F)
            except RuntimeError:
                res = 0.0
        return res

    def run(self, tasks, labels, K, epsilon=1e-5):
        n_ex = len(tasks)
        alpha = np.zeros(n_ex)
        duality_gap = 2 * epsilon

        idx_per_task = [[i for i in range(len(tasks)) if tasks[i] == s] for s in sorted(list(set(tasks)))]
        n_tasks = len(idx_per_task)
        step = 0
        while duality_gap > epsilon and step < self.max_iterations:  # TODO: set the duality_gap
            step += 1
            rand_i = np.random.randint(0, n_ex)
            if self.verbose > 0:
                print('Loop step: ', step, '| Example: ', rand_i)
            delta_i = self.solve_delta(K, alpha, tasks, labels, idx_per_task, rand_i)
            if self.verbose > 1:
                print('Delta: ', delta_i)
            alpha[rand_i] += delta_i

        self.omega = np.array([[np.float(((2 * self.k - 1) / (2 * self.k * self.lam))**(2 * self.k - 1) *
                                         (np.matrix(alpha[idx_per_task[s]]) *
                                          np.matrix(K[idx_per_task[s], :][:, idx_per_task[z]]) *
                                          np.matrix(alpha[idx_per_task[z]]).T)**(2 * self.k - 1))
                                for s in range(n_tasks)] for z in range(n_tasks)])  # Eq. (10)
        # print('Omega: \n', omega)

        self.beta = np.array([[alpha[i] * self.omega[s][tasks[i]] for s in range(n_tasks)]
                              for i in range(n_ex)])  # Eq. (7)
        # print('Shape beta', beta.shape) # examples x tasks
        # print('beta', beta)
        # return np.matrix(self.beta)

    def prediction_function(self, Ktest):
        return self.beta.T * np.matrix(Ktest)  # Eq. (5)

    def classify(self, Ktest):
        pred = self.prediction_function(Ktest)
        # print('Predictions \n', pred)
        # import matplotlib.pyplot as plt
        # plt.plot(np.array(pred).T, 'o')
        # plt.legend(range(len(pred)))
        # plt.show()
        return np.argmax(pred, 0).tolist()[0]


if __name__ == "__main__":
    # Example of usage

    # With iris
    IRIS = False
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

        y = [int(i[0]) for i in y.tolist()]

        k = 1
        lam = 1.0
        C = 1.0
        eff_output = EfficientOutputKernel(lam=lam, C=C, k=k, max_iterations=500)
        eff_output.run(K=K, tasks=y)

        classification = eff_output.classify(K)
        y_true = np.argmax(Y, axis=1)
        print([y for y in y_true])
        print('Class 0, #ex:', sum([1 if y == 0 else 0 for y in y_true]))
        print('Class 1, #ex:', sum([1 if y == 1 else 0 for y in y_true]))
        print('Class 2, #ex:', sum([1 if y == 2 else 0 for y in y_true]))
        print([y for y in classification][0])
        accuracy_train = accuracy_score([y for y in y_true], [y for y in classification][0])
        print('Accuracy training set:', accuracy_train)

    sanity = True
    if sanity:
        X = np.matrix([[1, 1, 0], [1, 2, 0], [-1, -1, 0], [-1, -3, 0], [0, 0, 2]])
        y = [0, 0, 1, 1, 2]
        K = X * X.T
        # print('K', K)

        k = 1
        lam = 1.0
        C = 1.0

        eff_output = EfficientOutputKernel(lam=lam, C=C, k=k, max_iterations=500)
        eff_output.run(K=K, tasks=y)
        classification = eff_output.classify(K)
        print('Classification train: \n', classification)
        print('Beta: \n', eff_output.get_beta())

        Xtest = np.matrix([[1, 1, 56], [-1, -1, 6]])

        classification = eff_output.classify(X * Xtest.T)
        print('Classification test: \n', classification)


