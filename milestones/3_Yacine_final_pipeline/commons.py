#!/usr/bin/env python

# General Import
import os
import sys

# Data science import
import joblib
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from math import floor

# sklearn imports
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import GridSearchCV

from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# File and Dir Path Config
DF_FILE_PATH = "/lustre03/project/6010672/yacine08/aec_vs_pli/result/features.csv" # "/home/yacine/Documents/BIAPT/features.csv"
OUTPUT_DIR = "/lustre03/project/6010672/yacine08/aec_vs_pli/result/" #"/home/yacine/Documents/BIAPT/testing/"

# Data Structures used in the analysis
EPOCHS = {
    "ec1": 1,
    "emf5": 2,
    "eml5": 3,
    "ec8": 4
}

GRAPHS = ["aec", "pli"]

FILTER_REGEX = {
    "func": "bin|wei",
    "wei": "std|mean|bin",
    "bin": "std|mean|wei",
    "func-wei": "bin",
    "func-bin": "wei"
}


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
    df = pd.read_csv(DF_FILE_PATH)

    # Keep only the Graph of interest
    df = df[df.graph == (GRAPHS.index(graph)+1)]

    # Keep only the epoch of interest
    df = df[(df.epoch == EPOCHS[epoch]) | (df.epoch == EPOCHS['ec1'])]


    # Keep only the features of interest
    df.drop(df.filter(regex=FILTER_REGEX[feature_group]), axis=1, inplace=True)

    print(f"Feature Group: {feature_group}:")
    print(df)

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
                 'clf__C': [0.1, 0.4, 0.5, 1, 10]},

                {'clf': [DecisionTreeClassifier()],  # Actual Estimator
                 'clf__criterion': ['gini', 'entropy']}]

def classify_loso(X, y, group, clf):
    """ Main classification function to train and test a ml model with Leave one subject out

        Args:
            X (numpy matrix): this is the feature matrix with row being a data point
            y (numpy vector): this is the label vector with row belonging to a data point
            group (numpy vector): this is the group vector (which is a the participant id)
            clf (sklearn classifier): this is a classifier made in sklearn with fit, transform and predict functionality

        Returns:
            f1s (list): the f1 at for each leave one out participant
    """
    logo = LeaveOneGroupOut()

    accuracies = []
    cms = np.zeros((2, 2))
    for train_index, test_index in logo.split(X, y, group):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        with joblib.parallel_backend('loky'):
            clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)

        acc = accuracy_score(y_test, y_hat)
        cm = confusion_matrix(y_test, y_hat)

        accuracies.append(acc)
        cms = np.add(cms, cm)
    return accuracies, cms


def classify_loso_model_selection(X, y, group, gs):
    """ This do classification using LOSO while also doing model selection using LOSO

        Args:
            X (numpy matrix): this is the feature matrix with row being a data point
            y (numpy vector): this is the label vector with row belonging to a data point
            group (numpy vector): this is the group vector (which is a the participant id)
            gs (sklearn GridSearchCV): this is a gridsearch object that will output the best model

        Returns:
            accuracies (list): the accuracy at for each leave one out participant
    """

    logo = LeaveOneGroupOut()

    accuracies = []
    f1s = []
    cms = []

    best_params = []

    num_folds = logo.get_n_splits(X, y, group) # keep track of how many folds left
    for train_index, test_index in logo.split(X, y, group):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        group_train, group_test = group[train_index], group[test_index]

        print(f"Number of folds left: {num_folds}")

        with joblib.parallel_backend('loky'):
            gs.fit(X_train, y_train, groups=group_train)

        y_hat = gs.predict(X_test)

        accuracy = accuracy_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat)
        cm = confusion_matrix(y_test, y_hat)

        accuracies.append(accuracy)
        f1s.append(f1)
        cms.append(cm)

        best_params.append(gs.best_params_)

        num_folds = num_folds - 1
    return accuracies, f1s, cms, best_params


def permutation_test(X, y, group, clf, num_permutation=1000):
    """ Helper function to validate that a classifier is performing higher than chance

        Args:
            X (numpy matrix): this is the feature matrix with row being a data point
            y (numpy vector): this is the label vector with row belonging to a data point
            group (numpy vector): this is the group vector (which is a the participant id)
            clf (sklearn classifier): this is a classifier made in sklearn with fit, transform and predict functionality
            num_permutation (int): the number of time to permute y
            random_state (int): this is used for reproducible output
        Returns:
            f1s (list): the f1 at for each leave one out participant

    """

    logo = LeaveOneGroupOut()
    train_test_splits = logo.split(X, y, group)

    with joblib.parallel_backend('loky'):
        (accuracies, permutation_scores, p_value) = permutation_test_score(clf, X, y, groups=group, cv=train_test_splits,
                                                                        n_permutations=num_permutation,
                                                                        verbose=num_permutation, n_jobs=-1)

    return accuracies, permutation_scores, p_value


def bootstrap_interval(X, y, group, clf, num_resample=1000, p_value=0.05):
    """Create a confidence interval for the classifier with the given p value

        Args:
            X (numpy matrix): The feature matrix with which we want to train on classifier on
            y (numpy vector): The label for each row of data point
            group (numpy vector): The group id for each row in the data (correspond to the participant ids)
            clf (sklearn classifier): The classifier that we which to train and validate with bootstrap interval
            num_resample (int): The number of resample we want to do to create our distribution
            p_value (float): The p values for the upper and lower bound

        Returns:
            f1_distribution (float vector): the distribution of all the f1s
            f1_interval (float vector): a lower and upper interval on the f1s corresponding to the p value
    """

    # Setup the pool of available cores
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    pool = mp.Pool(processes=ncpus)

    # Calculate each round asynchronously
    results = [pool.apply_async(bootstrap_classify, args=(X, y, group, clf, sample_id,)) for sample_id in range(num_resample)]

    # Unpack the results
    acc_distribution = [p.get() for p in results]

    # Sort the results
    acc_distribution.sort()

    # Set the confidence interval at the right index
    lower_index = floor(num_resample * (p_value / 2))
    upper_index = floor(num_resample * (1 - (p_value / 2)))
    acc_interval = acc_distribution[lower_index], acc_distribution[upper_index]

    return acc_distribution, acc_interval


# Create LOSO Grid Search to search amongst many classifier
class DummyEstimator(BaseEstimator):
    """Dummy estimator to allow gridsearch to test many estimator"""

    def fit(self): pass
    
    def score(self): pass


def create_gridsearch_pipeline():
    """ Helper function to create a gridsearch with a search space containing classifiers

        Returns:
            gs (sklearn gridsearch): this is the grid search objec wrapping the pipeline
    """

    # Create a pipeline
    pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', DummyEstimator())])  # Placeholder Estimator

    # Candidate learning algorithms and their hyperparameters
    search_space = [{'clf': [LogisticRegression()],  # Actual Estimator
                     'clf__penalty': ['l2'],
                     'clf__solver': ['lbfgs'],
                     'clf__max_iter': [1000],
                     'clf__C': np.logspace(0, 4, 10)},

                    {'clf': [LinearSVC()],
                     'clf__C': [1, 10, 100, 1000]},

                    {'clf': [DecisionTreeClassifier()],  # Actual Estimator
                     'clf__criterion': ['gini', 'entropy']}]
    

    # We will try to use as many processor as possible for the gridsearch
    gs = GridSearchCV(pipe, search_space, cv=LeaveOneGroupOut(), n_jobs=-1)
    return gs

def save_model(gs, model_file):
    
    model_file = open(model_file, 'ab')
    
    pickle.dump(gs, model_file)
    model_file.close()

def load_pickle(filename):
    '''Helper function to unpickle the pickled python obj'''
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    
    return data


def bootstrap_classify(X, y, group, clf, sample_id,): 
    print("Bootstrap sample #" + str(sample_id))
    sys.stdout.flush() # This is needed when we use multiprocessing

    # Get the sampled with replacement dataset
    sample_X, sample_y, sample_group = resample(X, y, group)

    # Classify and get the results
    accuracies, cms = classify_loso(sample_X, sample_y, sample_group, clf)

    return np.mean(accuracies)
