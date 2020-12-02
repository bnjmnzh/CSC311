from sklearn.impute import KNNImputer
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.transpose())
    acc = sparse_matrix_evaluate_by_item(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    # We use NaN-Euclidean distance measure.
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc

def sparse_matrix_evaluate_by_item(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_question_id, cur_user_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_question_id, cur_user_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def main():
    # sparse_matrix = load_train_sparse("../data").toarray()
    # val_data = load_valid_csv("../data")
    # test_data = load_public_test_csv("../data")
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")

    print("Sparse matrix:")
    print(sparse_matrix)
    # print("Shape of sparse matrix:")
    # print(sparse_matrix.shape) # (542, 1774)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    validation_accuracy = []
    for k in k_values:
        # acc = knn_impute_by_user(sparse_matrix, val_data, k)
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        validation_accuracy.append(acc)
    
    plt.plot(k_values, validation_accuracy, label='Validation Accuracy')
    plt.xlabel('K Values')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs K values')
    plt.show()
    best_index = np.argmax(validation_accuracy)
    best_k = k_values[best_index]
    # test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k)
    test_acc = knn_impute_by_item(sparse_matrix, test_data, best_k)
    print("k = {} had the highest validation accuracy and a test accuracy of {}.".format(best_k, test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()