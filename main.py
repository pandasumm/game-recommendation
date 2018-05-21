import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression


def dataCleaning(dataset):
    dataset.title = LabelEncoder().fit_transform(dataset.title)
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

    dataset = pd.read_csv("./games_added_publisher.csv")
    dataset.at[dataset['user_score'].abs() <= 5, 'user_score'] = 0
    dataset.at[dataset['user_score'].abs() > 5, 'user_score'] = 1

    resY = dataset.user_score
    dataX = dataCleaning(dataset)

    dataX.to_csv('./temp.csv', index=False)

    classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)

    classifier.fit(dataX, resY)
    accuracies = cross_val_score(estimator = classifier, X = dataX, y= resY, cv = 10)
    print("accuracies: ", accuracies.mean(), "+/-", accuracies.std(),"\n")
