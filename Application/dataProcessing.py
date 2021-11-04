import pandas as pd
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def clean_data():  # Specifiek script voor deze dataset, returned een dataset.
    data = pd.read_csv("sap_storing_data_hu_project.csv")
    storingen = pd.DataFrame(data=data, index=data.index,
                             columns=['stm_oorz_groep', 'stm_oorz_code', 'stm_geo_gst', 'stm_fh_ddt',
                                      'stm_sap_storeind_ddt', 'stm_sap_melddatum', 'stm_sap_meldtijd',
                                      'stm_aanngeb_dd', 'stm_aanngeb_tijd', 'stm_aanntpl_dd',
                                      'stm_aanntpl_tijd', 'stm_fh_dd', 'stm_fh_tijd', 'stm_fh_duur',
                                      'stm_sap_storeinddatum', 'stm_sap_storeindtijd',
                                      'stm_prioriteit'])
    storingen = storingen.iloc[1:, :]
    storingen = storingen.drop_duplicates()
    storingen = storingen.convert_dtypes()
    # Lege values priority 9 geven.
    storingen['stm_prioriteit'] = storingen['stm_prioriteit'].fillna(9)
    # Hieronder wordt een nieuwe code aangemaakt voor onbekende stm_ooz_codes, 0.
    storingen['stm_oorz_code'] = storingen['stm_oorz_code'].astype(str)
    storingen['stm_oorz_code'] = storingen['stm_oorz_code'].str.replace('<NA>', '0')
    storingen['stm_oorz_code'] = storingen['stm_oorz_code'].fillna('0')
    storingen['stm_oorz_code'] = storingen['stm_oorz_code'].astype(int)
    # Verwijderen van 0
    storingen = storingen[storingen.stm_fh_duur != 0]

    # Outlier cleanup
    duur = storingen['stm_fh_duur']
    q1 = duur.quantile(0.25)
    q3 = duur.quantile(0.75)
    IQR = q3 - q1
    storingen = storingen[storingen.stm_fh_duur < 3 * IQR]

    # Hieronder worden alle stm_geo_gst die niet numeriek zijn, verwijderd, zodat deze hele kolom kan worden omgezet tot een int.
    storingen = storingen[storingen.stm_geo_gst.astype(str).apply(lambda x: x.isnumeric())]
    storingen["stm_geo_gst"] = storingen["stm_geo_gst"].astype(int)

    # Hieronder worden alle tijden en data omgezet naar pure getallen, zodat het model hiermee kan werken.
    storingen["stm_sap_melddatum"] = pd.to_datetime(storingen["stm_sap_melddatum"])
    storingen["stm_sap_melddatum"] = storingen["stm_sap_melddatum"].dropna().apply(lambda a: int(a.strftime('%Y%m%d')))
    storingen["stm_sap_melddatum"] = storingen["stm_sap_melddatum"].astype(int)

    storingen["stm_sap_meldtijd"] = storingen["stm_sap_meldtijd"].str.replace(":", "")
    storingen = storingen[storingen.stm_sap_meldtijd != '']
    storingen["stm_sap_meldtijd"] = storingen["stm_sap_meldtijd"].dropna()
    storingen["stm_sap_meldtijd"] = storingen["stm_sap_meldtijd"].astype(int)

    storingen["stm_aanntpl_dd"] = storingen["stm_aanntpl_dd"].str.replace("/", "")
    storingen = storingen[storingen.stm_aanntpl_dd != '']
    storingen["stm_aanntpl_dd"] = storingen["stm_aanntpl_dd"].dropna()
    storingen["stm_aanntpl_dd"] = storingen["stm_aanntpl_dd"].astype(int)

    storingen["stm_aanntpl_tijd"] = storingen["stm_aanntpl_tijd"].str.replace(":", "")
    storingen = storingen[storingen.stm_aanntpl_tijd != '']
    storingen["stm_aanntpl_tijd"] = storingen["stm_aanntpl_tijd"].dropna()
    storingen["stm_aanntpl_tijd"] = storingen["stm_aanntpl_tijd"].astype(int)

    storingen["stm_aanngeb_dd"] = storingen["stm_aanngeb_dd"].str.replace("/", "")
    storingen = storingen[storingen.stm_aanngeb_dd != '']
    storingen["stm_aanngeb_dd"] = storingen["stm_aanngeb_dd"].dropna()
    storingen["stm_aanngeb_dd"] = storingen["stm_aanngeb_dd"].astype(int)

    storingen["stm_aanngeb_tijd"] = storingen["stm_aanngeb_tijd"].str.replace(":", "")
    storingen = storingen[storingen.stm_aanngeb_tijd != '']
    storingen["stm_aanngeb_tijd"] = storingen["stm_aanngeb_tijd"].dropna()
    storingen["stm_aanngeb_tijd"] = storingen["stm_aanngeb_tijd"].astype(int)

    # Hieronder worden dummies aangemaakt voor stm_oporzaak_groep.
    dummies = pd.get_dummies(storingen['stm_oorz_groep'])
    storingen = storingen.drop('stm_oorz_groep', axis=1)
    storingen = storingen.join(dummies)
    return storingen


def write_csv(data, name):
    data.to_csv(path_or_buf=name, index=False, mode='w', sep=';')


def train_test(storingen):
    # Allereerst moeten de x en de y worden bepaald.
    x = storingen[['stm_sap_meldtijd', 'stm_aanntpl_tijd']]
    y = storingen['stm_fh_duur']
    y = y.astype('int')

    # Daarna worden deze opgesplitst in een train en een test set.
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    return x_train, x_test, y_train, y_test


def decision_tree(depth, x_train, x_test, y_train, y_test):
    # Nu we onze data goed hebben voorbereid, kunnen er modellen mee getraind worden.
    # Allereerst trainen we een DecisionTreeClassifier, en bepalen we de score.
    boom = DecisionTreeClassifier(max_depth=depth)
    boom.fit(x_train, y_train)
    acc = boom.score(x_test, y_test)
    return acc, boom


def k_means(neighbours, x_train, x_test, y_train, y_test):
    # Nu trainen we ook een KNN model. Dit doen we ten eerste om te kijken welk model beter scoort, maar ook als een controle of ons eerste model niet overfit is.
    knn = KNeighborsClassifier(n_neighbors=neighbours)
    knn.fit(x_train, y_train)
    knn.set_params(weights="distance")
    acc = knn.score(x_test, y_test)
    # print("Model Accuracy: " + str(acc))
    return acc, knn


def prediction(targets, classifier):
    pred = classifier.predict([targets])
    return pred


def write_joblib(tree, kmeans):
    dump(tree, 'tree.joblib')
    dump(kmeans, 'kMeans.joblib')


def load_joblib():
    tree = load('tree.joblib')
    kmeans = load('kMeans.joblib')
    return tree, kmeans


write_csv(clean_data(), 'cleaned.csv')
# storingen = pd.read_csv("cleaned.csv",sep=';')
# X_train, X_test, Y_train, Y_test = trainTest(storingen)
# treeAcc, tree = decisionTree(52,X_train, X_test, Y_train, Y_test)
# kMeansAcc, kMeans = kMeans(55, X_train, X_test, Y_train, Y_test)
