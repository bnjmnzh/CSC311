from utils import *
from neural_network import *
from torch.autograd import Variable

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch

num_samples = 3

# TODO: complete this file.

def sub_sample(data):
    """ Generate a random assignment of rows. Return the new data.

    :param data: Dict
    :return: Dict
    """
    sample_length = data.shape[0] # 542
    indices = np.random.randint(sample_length, size=sample_length)
    resample = data[indices]

    return resample
    

def train_nn(model, lr, lamb, train_matrix, zero_train_matrix, test_data, num_epoch):
    """ Train the neural network on the training data and retunr the predictions on the test data.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param test_data: Dict
    :param num_epoch: int
    :return: Array
    """
    
    predictions = []

    train(model, lr, lamb, train_matrix, zero_train_matrix, test_data, num_epoch)

    for i, u in enumerate(test_data['is_correct']):
        user_id = test_data['user_id'][i]
        question_id = test_data['question_id'][i]
        inputs = Variable(zero_train_matrix[user_id]).unsqueeze(0)
        outputs = model(inputs)
        predictions.append(outputs[0][question_id].item())
    
    return predictions

def evaluate_predictions(predictions, test_data, threshold=0.5):
    """ Evaluate the test_data on the predictions. Return the test accuracy

    :param predictions: Array
    :param test_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :param threshold: float
    :return: float
    """
    correct = 0
    for i, u in enumerate(predictions):
        if u >= threshold and test_data['is_correct'][i]:
            correct += 1
        if u < threshold and not test_data['is_correct'][i]:
            correct += 1
    
    return correct / len(predictions)


def main():
    train_data = load_train_sparse('../data').toarray()
    valid_data = load_valid_csv('../data')
    test_data = load_public_test_csv('../data')

    parameters = {
        'lambd': [0.001,0.01,0.1],
        'k': [50, 50, 50]
    }

    predictions = []

    # Run neural network 
    for i in range(num_samples):
        
        # Set model hyperparameters
        k = parameters['k'][i]
        model = AutoEncoder(1774, k)

        # Set optimization hyperparameters.
        lr = 0.02
        num_epoch = 20
        lamb = parameters["lambd"][i]

        # Resample matrix
        train_matrix = sub_sample(train_data)

        # Generate Zero Train Matrix
        zero_train_matrix = train_matrix.copy()
        zero_train_matrix[np.isnan(train_matrix)] = 0

        zero_train_matrix = torch.FloatTensor(zero_train_matrix)
        train_matrix = torch.FloatTensor(train_matrix)

        # Train model
        print('Training model {}'.format(i))
        predictions.append(train_nn(model, lr, lamb, train_matrix, zero_train_matrix, test_data, num_epoch))
    
    # Average predictions of 3 models
    sum_predictions = np.sum(predictions, axis=0)
    average_predictions = [i/3 for i in sum_predictions]

    val_acc = evaluate_predictions(average_predictions, valid_data)
    print('Val Accuracy = {}'.format(val_acc))

    test_acc = evaluate_predictions(average_predictions, test_data)
    print('Test Accuracy = {}'.format(test_acc))


if __name__ == "__main__":
    main()
