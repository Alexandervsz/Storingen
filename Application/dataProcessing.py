import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


def cleanData():  # This code is not callable from the program itself. It exists here to create the desired dataset from console.
    data = pd.read_csv("sap_storing_data_hu_project.csv")
    disruptions = pd.DataFrame(data=data, index=data.index,
                               columns=['stm_oorz_groep', 'stm_oorz_code', 'stm_geo_gst', 'stm_fh_ddt',
                                        'stm_sap_storeind_ddt', 'stm_sap_melddatum', 'stm_sap_meldtijd',
                                        'stm_aanntpl_tijd', 'stm_fh_dd', 'stm_fh_tijd', 'stm_fh_duur',
                                        'stm_sap_storeinddatum', 'stm_sap_storeindtijd',
                                        'stm_prioriteit', 'stm_progfh_in_duur'])
    disruptions = disruptions.drop_duplicates()
    disruptions = pd.DataFrame(data=disruptions, index=disruptions.index,
                               columns=['stm_prioriteit', 'stm_aanntpl_tijd', 'stm_progfh_in_duur',
                                        'stm_sap_meldtijd', 'stm_fh_duur'])
    disruptions = disruptions.iloc[1:, :]
    disruptions = disruptions.convert_dtypes()
    disruptions['stm_prioriteit'] = disruptions['stm_prioriteit'].fillna(9)
    disruptions['stm_prioriteit'] = disruptions['stm_prioriteit'].astype(str)
    disruptions['stm_prioriteit'] = 'prio: ' + disruptions['stm_prioriteit']
    dummies = pd.get_dummies(disruptions['stm_prioriteit'])
    disruptions = disruptions.drop('stm_prioriteit', axis=1)
    disruptions = disruptions.join(dummies)
    disruptions = disruptions[disruptions.stm_fh_duur != 0]
    disruptions = disruptions[disruptions.stm_fh_duur < 1440]
    disruptions["stm_sap_meldtijd"] = disruptions["stm_sap_meldtijd"].str.replace(":", "")
    disruptions = disruptions[disruptions.stm_sap_meldtijd != '']
    disruptions["stm_sap_meldtijd"] = disruptions["stm_sap_meldtijd"].dropna()
    disruptions["stm_sap_meldtijd"] = disruptions["stm_sap_meldtijd"].astype(int)
    disruptions["stm_aanntpl_tijd"] = disruptions["stm_aanntpl_tijd"].str.replace(":", "")
    disruptions = disruptions[disruptions.stm_aanntpl_tijd != '']
    disruptions["stm_aanntpl_tijd"] = disruptions["stm_aanntpl_tijd"].dropna()
    disruptions["stm_aanntpl_tijd"] = disruptions["stm_aanntpl_tijd"].astype(int)
    disruptions = disruptions[disruptions.stm_progfh_in_duur.astype(str).apply(lambda x: not x.startswith("-"))]
    disruptions = disruptions[disruptions.stm_progfh_in_duur.astype(str).apply(lambda x: not x.endswith("-"))]
    disruptions = disruptions[disruptions.stm_progfh_in_duur.astype(str).apply(lambda x: x.isnumeric())]
    disruptions['stm_progfh_in_duur'] = disruptions['stm_progfh_in_duur'].astype(int)
    disruptions = disruptions[disruptions.stm_progfh_in_duur < 1440]
    disruptions = disruptions[disruptions.stm_progfh_in_duur > 0]

    return disruptions


def writeCSV(data, csvName):  # Writing a new .csv file with designate data and name.
    data.to_csv(path_or_buf=csvName, index=False, mode='w', sep=';')


def trainTest(storingen):
    # Allereerst moeten de x en de y worden bepaald.
    x = storingen[['stm_sap_meldtijd', 'stm_aanntpl_tijd']]
    y = storingen['stm_fh_duur']
    y = y.astype('int')

    # Daarna worden deze opgesplitst in een train en een test set.
    X_train, X_test, Y_train, Y_test = train_test_split(x, y)
    return X_train, X_test, Y_train, Y_test


def decisionTree(depth, X_train, X_test, Y_train, Y_test):
    # Nu we onze data goed hebben voorbereid, kunnen er modellen mee getraind worden.
    # Allereerst trainen we een DecisionTreeClassifier, en bepalen we de score.
    boom = DecisionTreeClassifier(max_depth=depth)
    boom.fit(X_train, Y_train)
    acc = boom.score(X_test, Y_test)
    # print("Acc: " + str(acc))

    # Hieronder staat het script dat de beste waarde voor de max depth opzoekt. We zijn hier geintereseerd in de waarde waar de curve afvlakt.

    return acc, boom


def kMeans(neighbours, X_train, X_test, Y_train, Y_test):
    # Nu trainen we ook een KNN model. Dit doen we ten eerste om te kijken welk model beter scoort, maar ook als een controle of ons eerste model niet overfit is.
    knn = KNeighborsClassifier(n_neighbors=neighbours)
    knn.fit(X_train, Y_train)
    knn.set_params(weights="distance")
    acc = knn.score(X_test, Y_test)
    # print("Model Accuracy: " + str(acc))
    return acc, knn


def polynomial():
    X = storingen[
        ['prio: 1', 'prio: 2', 'prio: 4', 'prio: 5', 'stm_sap_meldtijd', 'stm_aanntpl_tijd', 'stm_progfh_in_duur']]
    Y = storingen['stm_fh_duur'].astype('int')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    pf = PolynomialFeatures(degree=3)
    pf_ft = pf.fit_transform(X_test)
    pf.fit(pf_ft, Y_test)
    l_r = LinearRegression()
    l_r.fit(pf_ft, Y_test)
    acc = l_r.score(pf_ft, Y_test)

    return acc, pf, l_r


def prediction(lst, classifier):
    prediction = classifier.predict([lst])
    probability = classifier.predict_proba([lst])[-1]
    probability = probability.tolist()
    probability.sort()
    return prediction, probability[-1] * 100


def writeJobLib(tree, kMeans, poly, lr):
    dump(tree, 'tree.joblib')
    dump(kMeans, 'kMeans.joblib')
    dump(poly, 'poly.joblib')
    dump(lr, 'lr.joblib')


def loadJobLib():
    tree = load('tree.joblib')
    kMeans = load('kMeans.joblib')
    poly = load('poly.joblib')
    lr = load('lr.joblib')
    return tree, kMeans, poly, lr


# data = cleanData()
# writeCSV(data, 'cleaned.csv')
#
# storingen = pd.read_csv("cleaned.csv", sep=';')
# X_train, X_test, Y_train, Y_test = trainTest(storingen)
# treeAcc, tree = decisionTree(52, X_train, X_test, Y_train, Y_test)
# kMeansAcc, kMeans = kMeans(55, X_train, X_test, Y_train, Y_test)
# polyAcc, poly, lr = polynomial()
#
# writeJobLib(tree, kMeans, poly, lr)
