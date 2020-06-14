import pickle
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import config as cfg


# Create LOSO Grid Search to search amongst many classifier
class DummyEstimator(BaseEstimator):
    """Dummy estimator to allow gridsearch to test many estimator"""

    def fit(self): pass

    def score(self): pass


def print_summary(accuracies, group):
    """
    Helper function to print a summary of a classifier performance
    :param accuracies: a list of the accuracy obtained across fold (participant)
    :param group: ids of the participants windows
    :param df: dataframe containing all the data about the participant
    :return: None
    """

    p_ids = np.unique(group)
    print("Accuracies: ")
    for accuracy, p_id in zip(accuracies, p_ids):
        print(f"Participant {p_id}: accuracy = {accuracy}")

    print(f"Mean accuracy: {np.mean(accuracies)}")


def filter_dataframe(graph, epoch, feature_group):
    """ Helper function to filter the dataframe for a specific binary classifier"""

    # Read the CSV
    df = pd.read_csv(cfg.DF_FILE_PATH)

    # Keep only the Graph of interest
    df = df[df.graph == cfg.GRAPHS.index(graph)]

    # Keep only the epoch of interest
    df = df[(df.epoch == cfg.EPOCHS[epoch]) | (df.epoch == cfg.EPOCHS['ec1'])]

    # Keep only the features of interest
    df.drop(df.filter(regex=cfg.FILTER_REGEX[feature_group]), axis=1, inplace=True)

    # Set the up the feature matrix, the label vector and the group ids
    X = df.drop(['p_id', 'frequency', 'epoch', 'graph', 'window'], axis=1).to_numpy()
    y = df.epoch.to_numpy()
    group = df.p_id.to_numpy()

    return X, y, group


def load_pickle(filename):
    """Helper function to unpickle the pickled python obj"""
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()

    return data


def find_best_model(best_params):
    """ helper fo find best model given the best parameter """

    models_occurence = {}
    for param in best_params:

        clf = param['clf']
        if isinstance(clf, LogisticRegression):
            C = param['clf__C']
            key = f"log_{C}"
        elif isinstance(clf, LinearSVC):
            C = param['clf__C']
            key = f"svc_{C}"
        elif isinstance(clf, DecisionTreeClassifier):
            criterion = param['clf__criterion']
            key = f"dec_{criterion}"
        elif isinstance(clf, RandomForestClassifier):
            n_estimators = param['clf__n_estimators']
            max_depth = param['clf__max_depth']
            min_samples_split = param['clf__min_samples_split']
            min_samples_leaf = param['clf__min_samples_leaf']
            key = f"rand_{n_estimators}_{max_depth}_{min_samples_split}_{min_samples_leaf}"
        elif isinstance(clf, LinearDiscriminantAnalysis):
            solver = param['clf__solver']
            key = f"lda_{solver}"

        if key not in models_occurence:
            models_occurence[key] = 1
        else:
            models_occurence[key] = models_occurence[key] + 1

    for key, value in models_occurence.items():
        print(f"{key} : {value}")
    best_clf_params = max(models_occurence, key=models_occurence.get)

    content = best_clf_params.split('_')
    if content[0] == "log":
        C = float(content[1])
        clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, C=C)
    elif content[0] == "svc":
        C = float(content[1])
        clf = LinearSVC(C=C)
    elif content[0] == "dec":
        criterion = content[1]
        clf = DecisionTreeClassifier(criterion=criterion)
    elif content[0] == "rand":
        n_estimators = int(content[1])
        max_depth = int(float(content[2]))
        min_samples_split = int(content[3])
        min_samples_leaf = int(content[4])
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    elif content[0] == "lda":
        solver = content[1]
        clf = LinearDiscriminantAnalysis(solver=solver)

    return clf


# Candidate learning algorithms and their hyperparameters
search_space = [{'clf': [LogisticRegression()],
                 'clf__penalty': ['l2'],
                 'clf__solver': ['lbfgs'],
                 'clf__max_iter': [1000],
                 'clf__C': np.logspace(0, 4, 10)},

                {'clf': [LinearDiscriminantAnalysis()],
                 'clf__solver': ['svd', 'lsqr']},

                {'clf': [LinearSVC()],
                 'clf__C': [0.2, 0.5, 1, 10, 100]},

                {'clf': [DecisionTreeClassifier()],  # Actual Estimator
                 'clf__criterion': ['gini', 'entropy']}]

