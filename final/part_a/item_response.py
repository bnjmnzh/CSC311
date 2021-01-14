from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """

    ### VECTORIZED ###
    N, M = data.shape

    theta_m = theta.reshape(-1,1) * np.ones((N,M))
    beta_m = beta.reshape(-1, 1) * np.ones((M,N))
    m = data * (theta_m - np.transpose(beta_m)) - np.log(1 + np.exp(theta_m - np.transpose(beta_m)))
    log_lklihood = np.nansum(np.nansum(m, axis=1))
    return -log_lklihood

    ### OG VERSION ###
    # log_lklihood = 0.0
    # for i in range(sparse_matrix.shape[0]):
    #     for j in range(sparse_matrix.shape[1]):
    #         if np.isnan(sparse_matrix[(i, j)]):
    #             continue
    #         log_lklihood += sparse_matrix[(i,j)] * (theta[i] - beta[j]) - np.log(1+np.exp(theta[i] - beta[j]))
    # return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    ### OG ATTEMPT 1 ###

    # d_theta = 0.0
    # for i in range(sparse_matrix.shape[0]):
    #     for j in range(sparse_matrix.shape[1]):
    #         if np.isnan(sparse_matrix[(i, j)]):
    #             continue
    #         d_theta += sparse_matrix[(i,j)] - sigmoid(theta[i] - beta[j])
    # d_beta = -d_theta
    #
    # theta += lr * d_theta
    # beta -= lr * d_beta
    # return theta, beta

    ## NEW ATTEMPT ###

    N, M = data.shape
    m = np.zeros(data.shape)
    for i in range(N):
        for j in range(M):
            m[i][j] = theta[i] - beta[j]
    m = sigmoid(m)

    d_theta = np.nansum(data - m, axis=1)
    theta += lr * d_theta
    d_beta = np.nansum(data - m, axis=0)
    beta -= lr*d_beta
    return theta,beta


def irt(sparse_matrix, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.

    N, M = sparse_matrix.shape
    theta = np.zeros(N)
    beta = np.zeros(M)

    val_acc_lst = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(sparse_matrix, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {} \t Iteration: {}".format(neg_lld, score, i))
        theta, beta = update_theta_beta(sparse_matrix, lr, theta, beta)
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst

def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

# def make_sparse_matrix(data):
#     def_value = float("nan")
#     mat = [[def_value]*1774] * 542
#
#     for i in range(len(data["user_id"])):
#         stu_i = data["user_id"][i]
#         ques_i = data["question_id"][i]
#         is_correct = data["is_correct"][i]
#         mat[stu_i][ques_i] = is_correct
#     return np.array(mat)

# def main():
#     train_data = load_train_csv("../data")
#     # You may optionally use the sparse matrix.
#     sparse_matrix = load_train_sparse("../data")
#     val_data = load_valid_csv("../data")
#     test_data = load_public_test_csv("../data")
#
#
#     #np.set_printoptions(threshold=np.inf)
#
#     #print(make_sparse_matrix(val_data))
#     #print(sparse_matrix)
#
#     #####################################################################
#     # TODO:                                                             #
#     # Tune learning rate and number of iterations. With the implemented #
#     # code, report the validation and test accuracy.                    #
#     #####################################################################
#     #print("Validation Accuracy:", irt(sparse_matrix, val_data, 0.01, 4000))
#     #####################################################################
#
#     #####################################################################
#     # TODO:                                                             #
#     # Implement part (c)                                                #
#     #####################################################################
#     pass
#     #####################################################################
#     #                       END OF YOUR CODE                            #
#     #####################################################################


if __name__ == "__main__":
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    np.set_printoptions(threshold=np.inf)
    irt(sparse_matrix.toarray(), val_data, 0.001, 500)
