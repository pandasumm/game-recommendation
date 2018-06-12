import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression


def dataCleaning(dataset):
    dataset.title = LabelEncoder().fit_transform(dataset.title)
    dataset.developer = dataset.developer.fillna('N')
    dataset.developer = LabelEncoder().fit_transform(dataset.developer)
    dataset.rating = dataset.rating.fillna('E')
    dataset.rating = LabelEncoder().fit_transform(dataset.rating)
    dataset.platform = LabelEncoder().fit_transform(dataset.platform)
    dataset.publisher = LabelEncoder().fit_transform(dataset.publisher)


    genre = dataset.genre
    lenData = genre.shape[0]
    dataset = dataset.drop(['genre'], axis=1)

    genreSet = set()
    # print(genre.at[0])
    for i in range(lenData):
        for g in genre.at[i].split(','):
            genreSet.add(g)

    tData = {}
    for g in genreSet:
        tData[g] = [0 for i in range(lenData)]

    for i in range(lenData):
        for g in genre.at[i].split(','):
            tData[g][i] = 1

    newGenre = pd.DataFrame(data = tData)
    dataset = pd.concat([dataset, newGenre], axis=1)
    # print(dataset)
    return dataset


if __name__ == "__main__":

    dataset = pd.read_csv("./games_521.csv")
    dataset.at[dataset['user_score'].between(0,3,inclusive=False), 'user_score'] = 0
    dataset.at[dataset['user_score'].between(3,5,inclusive=False), 'user_score'] = 1
    dataset.at[dataset['user_score'].between(5,7,inclusive=False), 'user_score'] = 2
    dataset.at[dataset['user_score'].between(7,9,inclusive=False), 'user_score'] = 3
    dataset.at[dataset['user_score'].between(9,10), 'user_score'] = 3
    resY = dataset.user_score
    dataset = dataset.drop(['user_score'], axis = 1)

    dataX = dataCleaning(dataset)

    dataX.to_csv('./temp.csv', index=False)

    # 15-NN classifier
    classifier = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
    classifier.fit(dataX, resY)
    accuracies = cross_val_score(estimator = classifier, X = dataX, y= resY, cv = 10)
    print("15-nn classifier accuracies: ", accuracies.mean(), "+/-", accuracies.std(),"\n")


    # MLPClassifier, multi layer perceptron
    # classifier = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
    classifier = MLPClassifier(solver='lbfgs', alpha = 1e-5, hidden_layer_sizes=(5,2), random_state=1)
    classifier.fit(dataX, resY)
    accuracies = cross_val_score(estimator = classifier, X = dataX, y= resY, cv = 10)
    print("multi layer perceptron accuracies: ", accuracies.mean(), "+/-", accuracies.std(),"\n")
