from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################

    classifications = []
    k = [1, 3, 5, 7, 9]
    for i in k:
        predicted_valid = knn(i, train_inputs, train_targets, valid_inputs)
        valid_accuracy = 100 * np.sum(valid_targets == predicted_valid) / len(valid_targets)
        classifications.append(valid_accuracy)
    
    plt.plot(k, classifications, label="Classification Rate")
    plt.xlabel('K values')
    plt.ylabel('Classification Rate (%)')
    plt.title('Performance under different values of k')
    plt.show()

    max_index = np.argmax(classifications)
    best_k = k[max_index]
    print("k = {} had a classification rate of {} on the validation set".format(best_k, classifications[max_index]))
    print("k = {} had a classification rate of {} on the validation set".format(best_k - 2, classifications[max_index - 1]))
    print("k = {} had a classification rate of {} on the validation set".format(best_k + 2, classifications[max_index + 1]))
    predicted_test = knn(best_k, train_inputs, train_targets, test_inputs)
    test_acc = 100 * np.sum(test_targets == predicted_test) / len(test_targets)
    print("k ={} had a classification rate of {} on the test data".format(best_k, test_acc))
    predicted_test = knn(best_k - 2, train_inputs, train_targets, test_inputs)
    test_acc = 100 * np.sum(test_targets == predicted_test) / len(test_targets)
    print("k ={} had a classification rate of {} on the test data".format(best_k - 2, test_acc))
    predicted_test = knn(best_k + 2, train_inputs, train_targets, test_inputs)
    test_acc = 100 * np.sum(test_targets == predicted_test) / len(test_targets)
    print("k ={} had a classification rate of {} on the test data".format(best_k + 2, test_acc))
    

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
