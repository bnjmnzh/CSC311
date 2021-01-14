import numpy as np
import matplotlib.pyplot as plt

def select_knn_model(input_train, target_train, input_validate, target_validate, input_test, target_test):  
    accuracy_report = []  
    results = {}  
    for k in range(1, 21):  
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)  
        knn.fit(input_train, target_train)  
        predict_train = knn.predict(input_train)  
        train_accuracy = 100 * np.sum(predict_train == target_train) / len(target_train)  
        predict_validate = knn.predict(input_validate)  
        validate_accuracy = 100 * np.sum(predict_validate == target_validate) / len(target_validate)  
        print("k = {} has accuracy {}% on train and {}% on validate".format(k, train_accuracy, validate_accuracy))  
        accuracy_report.append((k, validate_accuracy))  
        results[k] = (100 - train_accuracy, 100 - validate_accuracy)  
    accuracy_report.sort(key =lambda tup: tup[1], reverse=True)  
    best_k = accuracy_report[0][0]  
    best_accuracy = accuracy_report[0][1]  
    print("k = {} has best accuracy with {}% accurate".format(best_k, best_accuracy))  
    print("-------------------------now testing on test set-------------------------")  
    knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)  
    knn.fit(input_train, target_train)  
    predict_test = knn.predict(input_test)  
    accuracy_test = 100 * np.sum(predict_test == target_test) / len(target_test)  
    print("k = {} has {}% accurate on test set".format(best_k, accuracy_test))  
    return results

def load_data():  
    fake_file = open("clean_fake.txt", "r")  
    real_file = open("clean_real.txt", "r")  
    data = fake_file.read().splitlines()  
    len_fake = len(data)  
    data.extend(real_file.read().splitlines())  
    len_real = len(data) - len_fake  
    vectorizer = CountVectorizer()  
    input_set = vectorizer.fit_transform(data).toarray()  
    target_set = np.concatenate((np.zeros(len_fake), np.ones(len_real)), axis = None)  
    feature_set = vectorizer.get_feature_names()  

      
    input_train, input_validate_test, target_train, target_validate_test = train_test_split(input_set, target_set, test_size=0.3, random_state=42)  
    input_validate, input_test, target_validate, target_test = train_test_split(input_validate_test, target_validate_test, test_size=0.5, random_state=42)  
    return input_train, input_validate, input_test, target_train, target_validate, target_test, feature_set


if __name__ == '__main__':
    x = range(10)
    y1 = np.cos(x)
    y2 = np.sin(x)

    plt.plot(x, y1, label='cosine')
    plt.plot(x, y2, label='sine')
    plt.legend()
    plt.xlabel('K values')
    plt.ylabel('Accuracy Score')
    plt.title('Performance Under Different K Values')
    plt.show()