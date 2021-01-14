from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

# Path to the data sets
real = 'clean_real.txt'
fake = 'clean_fake.txt'

""" Loads clean data and fake data sets and processess with CountVectorizer. Then split entire dataset into
70% training, 15% validation, and 15% test examples.
"""
def load_data():
    data = []
    with open(real, 'r') as file:
        real_data = file.read().split("\n")
        data += real_data
        num_real = len(real_data)

    with open(fake, 'r') as file:
        fake_data = file.read().split("\n")
        data += fake_data
        num_fake = len(fake_data)

    label = [0 for i in range(num_fake)]
    label = label + [1 for i in range(num_real)]

    vectorizer = CountVectorizer()
    fit = vectorizer.fit_transform(data)

    # Split the data into train, validate, test
    x_train, x_test_validate, y_train, y_test_validate = train_test_split(fit, label, train_size=.7, random_state=69)
    x_test, x_validate, y_test, y_validate = train_test_split(x_test_validate, y_test_validate, train_size=.5, random_state=69)

    return {"x_train": x_train,
            "y_train": y_train,
            "x_validate": x_validate,
            "y_validate": y_validate,
            "x_test": x_test,
            "y_test": y_test}

""" Uses KNN classifier to classify between real vs fake news. Use a range of k from 1 to 20, reporting
both training and validation errors.
"""
def select_knn_model(data_set):
    accuracy = []
    for k in range(1, 21):
        clf = neighbors.KNeighborsClassifier(n_neighbors = k, n_jobs = -1)
        clf.fit(data_set["x_train"], data_set["y_train"])
        train_score = clf.score(data_set["x_train"], data_set["y_train"])
        validate_score = clf.score(data_set["x_validate"], data_set["y_validate"])
        print("k = {} has error rate of {} on training and {} on validate".format(k, 1 - train_score, 1 - validate_score))
        accuracy.append((k, train_score, validate_score))

    train_score = []
    validate_score = []
    for i in accuracy:
        train_score.append(i[1])
        validate_score.append(i[2])

    accuracy.sort(key = lambda tup: tup[2], reverse=True)
    best_k = accuracy[0][0]
    best_err = accuracy[0][2]

    print("k = {} has the best accuracy on training of {}".format(best_k, best_err))
    print("k = {} has the best accuracy on validation of {}".format(best_k, best_err))

    best_clf = neighbors.KNeighborsClassifier(n_neighbors = best_k, n_jobs = -1)
    best_clf.fit(data_set["x_train"], data_set["y_train"])

    test_score = best_clf.score(data_set["x_test"], data_set["y_test"])
    print("k = {} has {} accuracy on the test set".format(best_k, test_score))

    # Plot training and validation error for each k
    plt.plot(range(1, 21), train_score, color='red', label='Training Accuracy')
    plt.plot(range(1, 21), validate_score, color='blue', label='Validate Accuracy')
    plt.legend()
    plt.xlabel('K values')
    plt.ylabel('Accuracy')
    plt.title('Performance Under Different K Values')
    plt.show()

""" Uses KNN classifier to classify between real vs fake news. Use a range of k from 1 to 20, reporting
both training and validation errors. Passes argument metric=‘cosine’ to the KNeighborsClassifier.
"""
def select_knn_model2(data_set):
    accuracy = []
    for k in range(1, 21):
        clf = neighbors.KNeighborsClassifier(n_neighbors = k, n_jobs = -1, metric='cosine')
        clf.fit(data_set["x_train"], data_set["y_train"])
        train_score = clf.score(data_set["x_train"], data_set["y_train"])
        validate_score = clf.score(data_set["x_validate"], data_set["y_validate"])
        print("k = {} has error rate of {} on training and {} on validate".format(k, 1 - train_score, 1 - validate_score))
        accuracy.append((k, train_score, validate_score))

    train_score = []
    validate_score = []
    for i in accuracy:
        train_score.append(i[1])
        validate_score.append(i[2])

    accuracy.sort(key = lambda tup: tup[2], reverse=True)
    best_k = accuracy[0][0]
    best_err = accuracy[0][2]

    print("k = {} has the best accuracy on training of {}".format(best_k, best_err))
    print("k = {} has the best accuracy on validation of {}".format(best_k, best_err))

    best_clf = neighbors.KNeighborsClassifier(n_neighbors = best_k, n_jobs = -1, metric='cosine')
    best_clf.fit(data_set["x_train"], data_set["y_train"])

    test_score = best_clf.score(data_set["x_test"], data_set["y_test"])
    print("k = {} has {} accuracy on the test set".format(best_k, test_score))

    # Plot training and validation error for each k
    plt.plot(range(1, 21), train_score, color='red', label='Training Accuracy')
    plt.plot(range(1, 21), validate_score, color='blue', label='Validate Accuracy')
    plt.legend()
    plt.xlabel('K values')
    plt.ylabel('Accuracy')
    plt.title('Performance Under Different K Values')
    plt.show()


if __name__ == '__main__':
    data_set = load_data()
    select_knn_model(data_set)
    select_knn_model2(data_set)