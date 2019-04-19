# SVC Implementation of model

from __future__ import print_function
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.decomposition import PCA

import data_loader
from sklearn.model_selection import train_test_split

batch_size = 1
num_classes = 10
epochs = 12


def train():
    # the data, split between train and test sets
    (data, labels) = data_loader.get_data()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=20)
    x_train = x_train.reshape(len(x_train), 256 * 256)
    x_test = x_test.reshape(len(x_test), 256 * 256)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    pca = PCA(n_components=50)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    svm = SVC()
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                   'C': [1, 10, 100, 1000]}]
    # {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    grid = GridSearchCV(svm, parameters, verbose=3)
    grid.fit(x_train[0:7000], y_train[0:7000])  # grid search learning the best parameters

    print(grid.best_params_)

    # Now we train the best estimator in the full dataset
    best_svm = grid.best_estimator_
    best_svm.fit(x_train, y_train)
    print("svm done")

    print("Testing")
    print("score: ", best_svm.score(x_test, y_test, ))
