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
#from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import permutation_test_score

from sklearn.utils import resample

best_model = SVC(max_iter=10000, kernel='linear', C=0.1)
# File and Dir Path Config
OUTPUT_DIR = "/home/lotte/projects/def-sblain/lotte/aec_vs_wpli/results/";
DF_FILE_PATH_01 = "/home/lotte/projects/def-sblain/lotte/aec_vs_wpli/results/features_step01.csv";
DF_FILE_PATH_10 = "/home/lotte/projects/def-sblain/lotte/aec_vs_wpli/results/features_step10.csv";

# Data Structures used in the analysis
EPOCHS = {
    "ec1": 1,
    "ind": 2,
    "emf5": 3,
    "eml5": 4,
    "ec8": 5
}

GRAPHS = ["aec", "pli"]

FILTER_REGEX = {"raw": "mean|std"}

def print_summary(accuracies, group, best_params= None):
    """
    Helper function to print a summary of a classifier performance
    :param accuracies: a list of the accuracy obtained across fold (participant)
    :param group: ids of the participants windows
    :param df: dataframe containing all the data about the participant
    :return: None
    """
    if best_params != None:
        p_ids = np.unique(group)
        print("Accuracies: ")
        for accuracy, p_id, best_param in zip(accuracies, p_ids, best_params):
            print(f"Participant {p_id}: accuracy = {accuracy} model:{best_param}")

    else:
        p_ids = np.unique(group)
        print("Accuracies: ")
        for accuracy, p_id in zip(accuracies, p_ids):
            print(f"Participant {p_id}: accuracy = {accuracy}")

    print(f"Mean accuracy: {np.mean(accuracies)}")


def filter_dataframe(graph, epoch, s):
    """ Helper function to filter the dataframe for a specific binary classifier"""

    # Read the CSV
    if s=='01':
        df = pd.read_csv(DF_FILE_PATH_01)
    if s=='10':
        df = pd.read_csv(DF_FILE_PATH_10)

    # Keep only the Graph of interest
    df = df[df.graph == (GRAPHS.index(graph)+1)]

    # Keep only the epoch of interest
    df = df[(df.epoch == EPOCHS[epoch]) | (df.epoch == EPOCHS['ec1'])]

    print(df.shape)
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

    f1s = []
    accuracies = []
    cms = np.zeros((2, 2))
    for train_index, test_index in logo.split(X, y, group):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        with joblib.parallel_backend('loky'):
            clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)

        f1 = f1_score(y_test, y_hat)
        acc = accuracy_score(y_test, y_hat)
        cm = confusion_matrix(y_test, y_hat)

        f1s.append(f1)
        accuracies.append(acc)
        cms = np.add(cms, cm)

    return accuracies, f1s, cms


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

    # NEW: F1 score not used only returns accuracy
    return np.mean(accuracies)
