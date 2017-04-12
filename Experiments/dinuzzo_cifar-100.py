from datasets import load_cifar10, Datasets, Dataset
import matplotlib.pyplot as plt
from algorithms.dinuzzo import DinuzzoOutputKernel
from sklearn.metrics import accuracy_score
import numpy as np
import time
from save_and_load import load_obj

# datasets = load_cifar100(one_hot=True, partitions=[0.01, 0.01])
datasets = load_obj('../Data/CIFAR-100/partitions/datasets_1003')

training_set_X = np.matrix(datasets.train.data)
training_set_Y = datasets.train.target
validation_set_X = np.matrix(datasets.validation.data)
validation_set_Y = datasets.validation.target
final_training_set_X = np.matrix(np.vstack((datasets.train.data, datasets.validation.data)))
final_training_set_Y = np.vstack((datasets.train.target, datasets.validation.target))
test_set_X = np.matrix(datasets.test.data)
test_set_Y = datasets.test.target
Ktr = training_set_X * training_set_X.T
list_of_lam = [2**n for n in range(-1, 9)]
delta = 1e-12
Ktrte = training_set_X * test_set_X.T
Ktrva = training_set_X * validation_set_X.T
Kftrte = final_training_set_X * test_set_X.T
Kftr = final_training_set_X * final_training_set_X.T

max_acc = 0.0
optimal_lam = list_of_lam[0]

time_start = time.time()
for lam in list_of_lam:
    dinuzzo = DinuzzoOutputKernel(lam=lam, delta=delta, max_iterations=2000, verbose=0)
    dinuzzo.run(training_set_Y, Ktr)

    val_classify = dinuzzo.classify(Ktrva)
    y_true = np.argmax(validation_set_Y, axis=1)
    accuracy_val = accuracy_score([y for y in y_true], [y for y in val_classify])
    print('Lambda:', lam, ' with validation accuracy:', accuracy_val)

    val_classify = dinuzzo.classify(Ktr)
    y_true = np.argmax(training_set_Y, axis=1)
    accuracy_train = accuracy_score([y for y in y_true], [y for y in val_classify])
    print('[train accuracy:', accuracy_train, ']')

    if accuracy_val >= max_acc:
        max_acc = accuracy_val
        optimal_lam = lam

print('Validated lambda:', optimal_lam)

time_end_validation = time.time() - time_start
dinuzzo = DinuzzoOutputKernel(lam=optimal_lam, delta=delta, max_iterations=2000, verbose=0)
dinuzzo.run(final_training_set_Y, Kftr)
time_end_train = time.time() - time_end_validation - time_start
classify = dinuzzo.classify(Kftrte)
y_true = np.argmax(test_set_Y, axis=1)
accuracy_test = accuracy_score([y for y in y_true], [y for y in classify])
time_end_classification = time.time() - time_end_train - time_start

L = np.array(dinuzzo.get_L())
print('Matrix of correlation among tasks L: \n', L)

print('Times')
print('Validation time required:', time_end_validation, '[with ', len(list_of_lam), ' different values of lambda]')
print('Final training time required:', time_end_train)
print('Classification time required:', time_end_classification)

classify = dinuzzo.classify(Kftr)
y_true = np.argmax(final_training_set_Y, axis=1)
accuracy_training = accuracy_score([y for y in y_true], [y for y in classify])

print('Train accuracy: ', accuracy_training)
print('Test accuracy: ', accuracy_test)

# Plots:
np.fill_diagonal(L, 0.0)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(L, interpolation='nearest', cmap=plt.cm.gray)
fig.colorbar(cax)
plt.title('Matrix of correlation among tasks L')
plt.show()
