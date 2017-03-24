# This file is an implementation of the algorithm by Jawanpuria et al.
# contained in the paper "Efficient Output Kernel Learning for Multiple Task"

import numpy as np
import scipy as sp
# import cvxopt


def solve_delta(K, k, lam, C, alpha, tasks, idx_per_task, i):
    a = K[i, i]
    R = K.shape[0]
    J = K.shape[1]
    n_tasks = len(idx_per_task)
    b = np.array([[np.sum([K[r, j] * alpha[j] for j in range(J) if tasks[j] == s])
                   for r in range(R)] for s in range(n_tasks)])

    c = np.array([[np.matrix(alpha[idx_per_task[s]]) * np.matrix(K[idx_per_task[s], :][:, idx_per_task[z]])
                   * np.matrix(alpha[idx_per_task[z]]).T for s in range(n_tasks)] for z in range(n_tasks)])
    eta = (lam / (C * (4 * k - 2))) * ((2 * k - 1) / (2 * k * lam))**(2 * k)

    def L_conj(v):  # Conjugate function of the loss
        return v**2 / 4.0 + v

    def F(delta):
        r = tasks[i]
        res = L_conj((-alpha[i] - delta) / C)
        res += float(eta * ((a * delta**2 + 2 * b[r, r] * delta + c[r, r])**(2 * k)))
        res += eta * (2 * np.sum([(b[r, s] * delta + c[r, s])**(2 * k)
                                  for r in range(n_tasks) for s in range(n_tasks) if s != r]))
        res += eta * np.sum([c[s, z]**(2 * k) for s in range(n_tasks) for z in range(n_tasks)])
        return res

    res = sp.optimize.minimize_scalar(F)
    return res['x']


def mtl_sdca(K, k, lam, C, tasks, epsilon=1e-5):
    n_ex = len(tasks)
    alpha = np.zeros(n_ex)
    duality_gap = 2 * epsilon
    max_iterations = 10000

    idx_per_task = [[i for i in range(len(tasks)) if tasks[i] == s] for s in sorted(list(set(tasks)))]
    n_tasks = len(idx_per_task)
    step = 0
    while duality_gap > epsilon and step < max_iterations:  # TODO: set the duality_gap
        step += 1
        rand_i = np.random.randint(0, n_ex)
        print('Loop step: ', step, '| Example: ', rand_i)
        delta_i = solve_delta(K, k, lam, C, alpha, tasks, idx_per_task, rand_i)
        print('Delta: ', delta_i)
        alpha[rand_i] += delta_i

    omega = [[((2 * k - 1) / (2 * k * lam)) * (np.matrix(alpha[idx_per_task[s]]) *
              np.matrix(K[idx_per_task[s], :][:, idx_per_task[z]]) * np.matrix(alpha[idx_per_task[z]]).T)**(2 * k - 1)
              for s in range(n_tasks)] for z in range(n_tasks)]  # Eq. (10)

    beta = [[alpha[i] * omega[s][tasks[i]] for s in range(n_tasks)] for i in range(n_ex)]  # Eq. (7)

    return np.matrix([[float(val) for val in row] for row in beta])


def prediction_function(beta, Ktest):
    return beta.T * np.matrix(Ktest)


def classify(beta, Ktest):
    pred = prediction_function(beta, Ktest)
    return np.argmax(pred, 0)


if __name__ == "__main__":
    # Example of usage

    # With iris
    IRIS = True
    if IRIS:
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

        y = [int(i[0]) for i in y.tolist()]

        k = 2
        lam = 150.0
        C = 1.0
        beta = mtl_sdca(K=K, k=k, lam=lam, C=C, tasks=y)

        classification = classify(beta, K)
        print(classification)

