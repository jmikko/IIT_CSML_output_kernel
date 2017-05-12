from datasets import load_cifar10, Datasets, Dataset
import matplotlib.pyplot as plt
from algorithms.efficient_MTL_SDCA import EfficientOutputKernel
from sklearn.metrics import accuracy_score
import numpy as np
import time
from save_and_load import load_obj

#datasets = load_cifar100(one_hot=True, partitions=[0.001, 0.001])
datasets = load_obj('../Data/CIFAR-100/partitions/datasets_100')

training_set_X = np.matrix(datasets.train.data)
training_set_Y = np.ones((training_set_X.shape[0], ))
training_set_tasks = np.argmax(datasets.train.target, axis=1)

validation_set_X = np.matrix(datasets.validation.data)
validation_set_Y = np.ones((validation_set_X.shape[0], ))
validation_set_tasks = np.argmax(datasets.validation.target, axis=1)

final_training_set_X = np.matrix(np.vstack((datasets.train.data, datasets.validation.data)))
final_training_set_Y = np.ones((final_training_set_X.shape[0], ))
final_training_set_tasks = np.argmax(np.vstack((datasets.train.target, datasets.validation.target)), axis=1)

test_set_X = np.matrix(datasets.test.data)
test_set_Y = np.ones((test_set_X.shape[0], ))
test_set_tasks = np.argmax(datasets.test.target, axis=1)

duplicate = True
if duplicate:
    number_of_tasks = len(set(test_set_tasks))
    training_set_Y_tmp = [np.ones((training_set_X.shape[0], )) for i in range(number_of_tasks)]
    training_set_X_new = training_set_X
    for i in range(number_of_tasks - 1):
        training_set_X_new = np.vstack((training_set_X_new, training_set_X))
    training_set_X = np.matrix(training_set_X_new)
    training_set_Y_tmp2 = []
    for nrow, row in enumerate(training_set_Y_tmp):
        training_set_Y_tmp2.append([1.0 if training_set_tasks[nel] == nrow else 0.0 for nel, el in enumerate(row)])
    training_set_Y = []
    for row in training_set_Y_tmp2:
        training_set_Y = np.hstack((training_set_Y, row))
    training_set_Y = np.array(training_set_Y)
    training_set_tasks_tmp = np.array([[j for ntr in range(len(training_set_tasks))] for j in range(number_of_tasks)])
    training_set_tasks = []
    for row in training_set_tasks_tmp:
        training_set_tasks = np.hstack((training_set_tasks, row))
    training_set_tasks = np.array(training_set_tasks, dtype=int)

    final_training_set_Y_tmp = [np.ones((final_training_set_X.shape[0], )) for i in range(number_of_tasks)]
    final_training_set_X_new = final_training_set_X
    for i in range(number_of_tasks - 1):
        final_training_set_X_new = np.vstack((final_training_set_X_new, final_training_set_X))
    final_training_set_X = np.matrix(final_training_set_X_new)
    final_training_set_Y_tmp2 = []
    for nrow, row in enumerate(final_training_set_Y_tmp):
        final_training_set_Y_tmp2.append([1.0 if final_training_set_tasks[nel] == nrow else 0.0 for nel, el in enumerate(row)])
    final_training_set_Y = []
    for row in final_training_set_Y_tmp2:
        final_training_set_Y = np.hstack((final_training_set_Y, row))
    final_training_set_Y = np.array(final_training_set_Y)
    final_training_set_tasks_tmp = np.array([[j for ntr in range(len(final_training_set_tasks))] for j in range(number_of_tasks)])
    final_training_set_tasks = []
    for row in final_training_set_tasks_tmp:
        final_training_set_tasks = np.hstack((final_training_set_tasks, row))
    final_training_set_tasks = np.array(final_training_set_tasks, dtype=int)

list_of_lam = [0.5]  # [2**n for n in range(-1, 10)]
list_of_C = [16.0]  # [2**n for n in range(-1, 10)]
list_of_lam_C = [(l, c) for l in list_of_lam for c in list_of_C]

k = 1
#Ktr = training_set_X * training_set_X.T
#Ktrte = training_set_X * test_set_X.T
#Ktrva = training_set_X * validation_set_X.T
#Kftrte = final_training_set_X * test_set_X.T
Kftr = final_training_set_X * final_training_set_X.T

max_acc = 0.0
optimal_lam = list_of_lam_C[0][0]
optimal_C = list_of_lam_C[0][1]

time_start = time.time()
for (lam, C) in list_of_lam_C:
    efficient = EfficientOutputKernel(lam=lam, C=C, k=k, max_iterations=2, verbose=1)
    efficient.run(training_set_tasks, training_set_Y, Ktr)

    val_classify = efficient.classify(Ktrva)
    y_true = validation_set_tasks
    accuracy_val = accuracy_score([y for y in y_true], [y for y in val_classify])
    print('Lambda, C:', lam, C, ' with validation accuracy:', accuracy_val)

    # val_classify = efficient.classify(Ktr)
    # y_true = training_set_tasks
    # accuracy_train = accuracy_score([y for y in y_true], [y for y in val_classify])
    # print('[train accuracy:', accuracy_train, ']')

    if accuracy_val >= max_acc:
        max_acc = accuracy_val
        optimal_lam = lam
        optimal_C = C

print('Validated lambda, C:', optimal_lam, optimal_C)

time_end_validation = time.time() - time_start
efficient = EfficientOutputKernel(lam=optimal_lam, C=optimal_C, k=k, max_iterations=10000, verbose=1)
efficient.run(final_training_set_tasks, final_training_set_Y, Kftr)
time_end_train = time.time() - time_end_validation - time_start
classify = efficient.classify(Kftrte)
y_true = test_set_tasks
accuracy_test = accuracy_score([y for y in y_true], [y for y in classify])
time_end_classification = time.time() - time_end_train - time_start

print('Times')
print('Validation time required:', time_end_validation, '[with ', len(list_of_lam), ' different values of lambda]')
print('Final training time required:', time_end_train)
print('Classification time required:', time_end_classification)

classify = efficient.classify(Kftr)
print('Classification: \n', classify)
y_true = final_training_set_tasks
print('True labels: \n', y_true)
# accuracy_training = accuracy_score([y for y in y_true], [y for y in classify])

omega = np.array(efficient.get_omega())
print('Matrix Omega among tasks: \n', omega)

# print('Train accuracy: ', accuracy_training)
print('Test accuracy: ', accuracy_test)

dict_ID_name = datasets[2].general_info_dict['dict_ID_name']
print('Correspondence ID_class - Class:\n', dict_ID_name)
# Plots:
np.fill_diagonal(omega, 0.0)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(omega, interpolation='nearest', cmap=plt.cm.gray)
fig.colorbar(cax)
plt.title('Matrix of correlation among tasks L')
alpha = ['' + dict_ID_name[2 * k] + ' ' + dict_ID_name[2 * k + 1] for k in range(5)]
ax.set_xticklabels([''] + alpha)
ax.set_yticklabels([''] + alpha)
plt.show()
