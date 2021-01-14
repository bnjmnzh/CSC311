from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    hyperparameters = {
        "learning_rate": 0.001,
        "weight_regularization": 0.,
        "num_iterations": 4000
    }

    weights = np.zeros((M + 1, 1))
    run_check_grad(hyperparameters)

    train_entropy = []
    validate_entropy = []
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        weights = np.subtract(weights, hyperparameters['learning_rate'] * df)
        
        train_entropy.append(f)
        validate_ce = evaluate(valid_targets, logistic_predict(weights, valid_inputs))[0]
        validate_entropy.append(validate_ce)
    
    plt.figure()
    plt.plot(range(hyperparameters['num_iterations']), train_entropy, label='Training Cross Entropy')
    plt.plot(range(hyperparameters['num_iterations']), validate_entropy, label='Validate Cross Entropy')
    plt.title('mnist small, learning rate = {}, {} iterations, plot of cross entropy'.format(hyperparameters['learning_rate'], hyperparameters['num_iterations']))
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.show()

    print('Learning: {}, number of iterations: {}'.format(hyperparameters['learning_rate'], hyperparameters['num_iterations']))

    train_results = evaluate(train_targets, logistic_predict(weights, train_inputs))
    print("MNIST small training cross entropy: {}".format(train_results[0]))
    print("MNIST small training classification error: {}".format(1 - train_results[1]))

    validate_result = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
    print("MNIST small validate cross entropy: {}".format(validate_result[0]))
    print("MNIST validate classification error: {}".format(1 - validate_result[1]))

    test_result = evaluate(test_targets, logistic_predict(weights, test_inputs))
    print("MNIST small test cross entropy: {}".format(test_result[0]))
    print("MNIST small test classification error: {}".format(1 - test_result[1]))


def run_pen_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Implement the function that automatically evaluates different     #
    # penalty and re-runs penalized logistic regression 5 times.        #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.001,
        "weight_regularization": 0.,
        "num_iterations": 2000
    }

    lambdas = [0, 0.001, 0.01, 0.1, 1.0]

    
    for lmbda in lambdas:
        hyperparameters['weight_regularization'] = lmbda
        train_errors = 0
        validate_errors = 0 
        train_entropy = 0 # For averaged cross entropy
        validate_entropy = 0 # For averaged cross entropy
        training_entropy = []  # For reporting how training progresses
        validation_entropy = [] # For reporting how trianing progresses
        for j in range(5):
            weights = np.zeros((M + 1, 1))
            for t in range(hyperparameters['num_iterations']):
                f, df, y = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                weights = np.subtract(weights, hyperparameters['learning_rate'] * df)
        
                if j == 4:
                    training_entropy.append(f)
                    validate_entropy = evaluate(valid_targets, logistic_predict(weights, valid_inputs))[0]
                    validation_entropy.append(validate_entropy)
            
            if j == 4:
                plt.figure()
                plt.plot(range(hyperparameters['num_iterations']), training_entropy, label='Training cross entropies')
                plt.plot(range(hyperparameters['num_iterations']), validation_entropy, label='Validation cross entropies')
                plt.xlabel('Number of Iterations')
                plt.ylabel('Cross Entropy')
                plt.title('MNIST Small Number of iterations vs Cross Entropies for lambda = {}'.format(lmbda))
                plt.legend()
                plt.show()
            
            train_results = evaluate(train_targets, logistic_predict(weights, train_inputs))
            validate_results = evaluate(valid_targets, logistic_predict(weights, valid_inputs))

            train_entropy += train_results[0]
            train_errors += 1 - train_results[1]

            validate_entropy += validate_results[0]
            validate_errors += 1 - validate_results[1]

        train_entropy /= 5
        train_errors /= 5
        validate_entropy /= 5
        validate_errors /= 5
        
        print('Learning: {}, number of iterations: {}'.format(hyperparameters['learning_rate'], hyperparameters['num_iterations']))
        print('Training cross entropy for lambda = {}: {}'.format(lmbda, train_entropy))
        print('Training classificatio error for lambda = {}: {}'.format(lmbda, train_errors))
        print('Validation cross entropy for lambda = {}: {}'.format(lmbda, validate_entropy))
        print('Validation classificatio error for lambda = {}: {}'.format(lmbda, validate_errors))

        if lmbda == 0.1:
            test_results = evaluate(test_targets, logistic_predict(weights, test_inputs))
            print('lambda = {} has a cross entropy of {} on the test data'.format(lmbda, test_results[0]))
            print('lambda = {} has a classification error rate of {} on the test data'.format(lmbda, 1 - test_results[1]))


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


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
    # run_logistic_regression()
    run_pen_logistic_regression()
