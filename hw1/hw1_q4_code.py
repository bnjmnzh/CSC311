import numpy as np
import matplotlib.pyplot as plt

data_train = {
    'X': np.genfromtxt('data_train_X.csv', delimiter=','), 
    't': np.genfromtxt('data_train_y.csv', delimiter=',')
    }

data_test = {
    'X': np.genfromtxt('data_test_X.csv', delimiter=','),
    't': np.genfromtxt('data_test_y.csv', delimiter=',')
    }

""" 
Takes data as an argument, returns the randomly permtuted version along the samples.
"""
def shuffle_data(data):
    target = data[0]
    x = data[1]

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    target = target[indices]
    x = x[indices]
    
    return (target, x)
    
"""
Takes data, number of partitions as num_folds, and selected partition fold as arguments. Returns the selected 
partition fold as data_fold and remaining data as data_rest
num_folds, fold are both positive integers, 1 < fold < num_folds
"""
def split_data(data, num_folds, fold):
    target = np.array_split(data[0], num_folds)
    x = np.array_split(data[1], num_folds)

    data_fold = (target[fold], x[fold])

    new_target = np.concatenate((target[:fold] + target[fold + 1:]))
    new_x = np.concatenate((x[:fold] + x[fold + 1:]))

    data_rest = (new_target, new_x)

    return data_fold, data_rest


"""
Takes data and lambd as arguments, return coefficients of ridge regression with penalty level lambd
Ignore bias parameter for simplicity. 
"""
def train_model(data, lambd):
    t = data[0]
    X = data[1]
    N = X.shape[0]

    w = np.matmul(X.transpose(), X)
    id = lambd * N * np.identity(w.shape[0])

    w = np.linalg.inv(w + id)
    w = np.matmul(w, X.transpose())
    return np.matmul(w, t)


"""
Takes data and model as arugments and returns the average squared error loss based on model. If data is composed of 
t in R^n and X in R^{N x D}, model is w then the return value is ||Xw - t||^2 / 2N
"""
def loss(data, model):
    N = data[1].shape[0]
    err = np.matmul(data[1], model) - data[0]
    error = np.matmul(err.transpose(), err) / (2 * N)
    
    return error

"""
data is a (t, X) pair where t is response and X is feature matrix
"""
def cross_validation(data, num_folds, lambd_seq):
    cv_error = []
    data = shuffle_data(data)
    for i in range(0, len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1, num_folds):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error


if __name__ == '__main__':
    lambd_seq = np.linspace(0.00005, 0.005, num=50)

    train_errors = []
    test_errors = []
    for i in range(len(lambd_seq)):
        model = train_model((data_train['t'], data_train['X']), lambd_seq[i])
        train_errors.append(loss((data_train['t'], data_train['X']), model))
        test_errors.append(loss((data_test['t'], data_test['X']), model))
    
    print("training errors: ")
    for i in range(len(train_errors)):
        print("lambda = {}: {}".format(lambd_seq[i], train_errors[i]))
    
    print("test errors: ")
    for i in range(len(test_errors)):
        print("lambda = {}: {}".format(lambd_seq[i], test_errors[i]))

    cv_5 = cross_validation((data_train['t'], data_train['X']), 5, lambd_seq)
    cv_10 = cross_validation((data_train['t'], data_train['X']), 10, lambd_seq)

    min_5 = np.argmin(cv_5)
    min_10 = np.argmin(cv_10)

    lambda_5 = lambd_seq[min_5]
    lambda_10 = lambd_seq[min_10]

    print("Lambda = {} is the best one using 5 folds.".format(lambda_5))
    print("Lambda = {} is the best one using 10 folds.".format(lambda_10))

    plt.plot(lambd_seq, train_errors, label='Training Errors')
    plt.plot(lambd_seq, test_errors, label='Test Errors')
    plt.plot(lambd_seq, cv_5, label='CV5')
    plt.plot(lambd_seq, cv_10, label='CV10')
    plt.title('Plot of lambdas and errors')
    plt.legend()
    plt.xlabel('lambd_seq')
    plt.ylabel('Errors')
    plt.show()




