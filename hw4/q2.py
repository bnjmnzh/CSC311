'''
Question 2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        index = np.where(train_labels == i)[0]
        data = train_data[index]
        for j in range(64):
            means[i][j] = np.mean(data[:, j])
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        index = np.where(train_labels == i)[0]
        diff = np.zeros((64, 64))
        for j in index:
            a = train_data[int(j)] - means[int(i)]
            m_a = np.reshape(a, (1, 64))
            diff = diff + m_a.T.dot(m_a)
        covariances[i] = covariances[i] + diff / 700
        covariances[i] = covariances[i] + np.identity(64) / 100
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    likelihood = np.zeros((len(digits), 10))
    first = -32 * np.log((2 * np.pi))
    for i in range(len(digits)):
        for index in range(len(covariances)):
            det_cov = np.linalg.det(covariances[index])
            second = -0.5 * np.log(det_cov)
            diff = digits[i] - means[index]
            inv_cov = np.linalg.inv(covariances[index])
            third = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
            likelihood[i][index] = first + second + third
    return likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    likelihood = generative_likelihood(digits, means, covariances)
    cond_likelihood = np.zeros((len(digits), 10))
    alpha = np.log(0.1)
    for i in range(len(digits)):
        total_probability = np.sum(likelihood[i])
        for c in range(10):
            cond_likelihood[i][c] = likelihood[i][c] + alpha - total_probability

    return cond_likelihood

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    total_cond_like = len(cond_likelihood)
    b = 0
    for i in range(total_cond_like):
        get_labels = int(labels[i])
        b += cond_likelihood[i][get_labels]
    return b / total_cond_like

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    prediction = np.zeros(len(digits))
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    for n in range(len(prediction)):
        prediction[n] = np.argmax(cond_likelihood[n])
    return prediction

def evaluate_data(predictions, labels):
    N = len(predictions)
    count = 0
    for i in range(N):
        if predictions[i] == labels[i]:
            count += 1

    return count / N

def plot_covariances(covariances):
    nums = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        nums.append(np.log(cov_diag).reshape(8, 8))
    
    nums_concat = np.concatenate(nums, 1)
    plt.imshow(nums_concat, cmap='gray')
    plt.savefig('diagonal.png')
    plt.show()

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    print('Finding mles')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    # Find avg conditional likelihood on train and test data
    print('Finding liklihoods')

    train_avg_cond_log = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg_cond_log = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print('Average conditional log-likelihood on train data is {}'.format(train_avg_cond_log))
    print('Average conditional log-likelihood on test data is {}'.format(test_avg_cond_log))
    
    # Find accuracy on test and trains set
    # train_accuracy = evaluate_data(classify_data(train_data, means, covariances), train_labels)
    # test_accuracy = evaluate_data(classify_data(test_data, means, covariances), test_labels)
    # print('Accuracy on train set is {}'.format(train_accuracy))
    # print('Accuracy on test set is {}'.format(test_accuracy))

    # plot_covariances(covariances)


if __name__ == '__main__':
    main()
