# CHANGE THESE IMPORTS BEFORE PASSING IN
# from q2.check_grad import check_grad
# from q2.utils import *
# from q2.logistic import *

from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace


def plot_data(iteration, train_cross_entropies, val_cross_entropies, lmbd=None):
    plt.figure()
    plt.plot(iteration, train_cross_entropies, color='blue',
             label='Training Data Small Cross Entropies')
    plt.plot(iteration, val_cross_entropies, color='red',
             label='Validation Data Cross Entropies')
    plt.xlabel('Iterations')
    plt.ylabel('Cross Entropy')
    if lmbd:
        plt.title(
            'Training Data Small Cross Entropies Over Iterations For Lambda=' + lmbd)
    else:
        plt.title('Training Data Cross Entropies Over Iterations')
    plt.legend(loc="center right")
    plt.show()
    # plt.xticks(np.arange(1, 100, 100))
    # plt.yticks(np.arange(0, 1, 0.1))


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    hyperparameters = {
        "learning_rate": 0.12,
        "weight_regularization": 0.,
        "num_iterations": 1000
    }

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    # run_check_grad(hyperparameters)
    print('HYPERPARAMETER: ', hyperparameters['learning_rate'])
    alpha = hyperparameters['learning_rate']

    weights = np.zeros((M + 1, 1))
    f = df = y = None
    training_results = []
    valid_results = []

    for i in range(hyperparameters['num_iterations']):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        weights = np.subtract(weights, alpha*df)
        training_results.append(f)
        valid_res = evaluate(valid_targets,
                             logistic_predict(weights, valid_inputs))
        valid_results.append(valid_res[0])

    plot_data([i for i in range(1, hyperparameters['num_iterations'] + 1)],
              training_results, valid_results)

    result = evaluate(train_targets, logistic_predict(weights, train_inputs))
    print("MNIST Train Cross Entropy: ", result[0])
    print("MNIST Train Classification Rate: ", result[1])

    result = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
    print("Validation Data on MNIST Train Cross Entropy: ", result[0])
    print("Validation Data on MNIST Train Classification Rate: ",
          result[1])

    result = evaluate(test_targets, logistic_predict(weights, test_inputs))
    print("Test Data on MNIST Train Cross Entropy: ", result[0])
    print("Test Data on MNIST Train Classification Rate: ",
          result[1])


def run_pen_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    lambdas = [0, 0.001, 0.01, 0.1, 1]

    hyperparameters = {
        "learning_rate": 0.002,
        "weight_regularization": 0.,
        "num_iterations": 1000
    }

    alpha = hyperparameters['learning_rate']

    for lmbd in lambdas:
        hyperparameters['weight_regularization'] = lmbd
        train_cross_entropies = train_classification_errors = \
            valid_cross_entropies = valid_classification_errors = 0
        training_results = []
        valid_results = []

        for iteration in range(5):
            weights = np.zeros((M + 1, 1))
            f = df = y = None

            for i in range(hyperparameters['num_iterations']):
                f, df, y = logistic_pen(weights, train_inputs, train_targets,
                                        hyperparameters)
                weights = np.subtract(weights, alpha * df)
                if iteration == 3:
                    training_results.append(f)
                    valid_res = evaluate(valid_targets,
                                         logistic_predict(weights, valid_inputs))
                    valid_results.append(valid_res[0])

            if iteration == 3:
                plot_data([i for i in range(1, hyperparameters['num_iterations'] + 1)],
                          training_results, valid_results, lmbd=str(lmbd))

            train_res = evaluate(train_targets, logistic_predict(weights, train_inputs))
            valid_res = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
            train_cross_entropies += train_res[0]
            train_classification_errors += train_res[1]
            valid_cross_entropies += valid_res[0]
            valid_classification_errors += valid_res[1]

        train_cross_entropies /= 5
        train_classification_errors /= 5
        valid_cross_entropies /= 5
        valid_classification_errors /= 5

        print()
        print("Cross entropy training for lambda " + str(lmbd) + ":",
              train_cross_entropies)
        print("Classification errors training for lambda " + str(lmbd) + ":",
              train_classification_errors)
        print("Cross entropy validation for lambda " + str(lmbd) + ":",
              valid_cross_entropies)
        print("Classification errors validation for lambda " + str(lmbd) + ":",
              valid_classification_errors)


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
    # run_pen_logistic_regression()
