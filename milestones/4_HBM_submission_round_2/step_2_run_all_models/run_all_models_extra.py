"""
Charlotte Maschke November 11 2020
This script is to run several machine learnin models over a selection of different hyperparameters.
It does not use sklearn grid search, as the number of model parameters is not very large.
It will output several files with accuracies for the independent models, which can be summarized and visualized
using visualize_models.py.
"""
#

# General Import
import os
import sys

# Data science import
import pickle
import numpy as np

# Sklearn import
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#import models
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Add the directory containing your module to the Python path (wants absolute paths)
scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))

# Common import shared across analysis
import commons
from commons import classify_loso, print_summary
from commons import load_pickle, filter_dataframe,filter_dataframe_multiple

# This will be given by the srun in the bash file
# Get the argument
GRAPHS = ["aec", "pli", "both"]
Steps = ['01', '10']

# select hyperparameters
Cs= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
kernels = ['linear']
for s in Steps:
    for c in Cs:
        for k in kernels:
            for graph in GRAPHS:
                clf = SVC(max_iter=10000, kernel=k, C=c)

                # Run responsiveness vs. unresponsiveness
                print(f"Model: Graph {graph} at resp_vs_unre_step_{s}")

                if graph != "both":
                    print(f"MODE {graph}")
                    print(f"FINAL Model Graph {graph} at resp_vs_unres")
                    # responsive
                    X_r, y_r, group_r = filter_dataframe_multiple(graph, 'ec1', 'ec8', 'ind', s)
                    y_r[:] = 1
                    # unresponsive
                    X_u, y_u, group_u = filter_dataframe(graph, 'emf5', 'eml5', s)
                    y_u[:] = 4  # randomly choosen integer just need to be different from 1

                    # add responsive and unresponsive data together
                    X = np.vstack((X_r, X_u))
                    group = np.hstack((group_r, group_u))
                    y = np.hstack((y_r, y_u))

                if graph == "both":
                    print(f"MODE {graph}")
                    print(f"FINAL Model Graph {graph} at resp vs unres")
                    # responsive
                    X_r_aec, y_r_aec, group_r_aec = filter_dataframe_multiple('aec', 'ec1', 'ec8', 'ind', s)
                    y_r_aec[:] = 1
                    # unresponsive
                    X_u_aec, y_u_aec, group_u_aec = filter_dataframe('aec', 'emf5', 'eml5', s)
                    y_u_aec[:] = 4  # randomly choosen integer just need to be different from 1

                    X_aec = np.vstack((X_r_aec, X_u_aec))
                    group_aec = np.hstack((group_r_aec, group_u_aec))
                    y_aec = np.hstack((y_r_aec, y_u_aec))

                    # responsive
                    X_r_pli, y_r_pli, group_r_pli = filter_dataframe_multiple('pli', 'ec1', 'ec8', 'ind', s)
                    y_r_pli[:] = 1

                    # unresponsive
                    X_u_pli, y_u_pli, group_u_pli = filter_dataframe('pli', 'emf5', 'eml5', s)
                    y_u_pli[:] = 4  # randomly choosen integer just need to be different from 1

                    X_pli = np.vstack((X_r_pli, X_u_pli))
                    group_pli = np.hstack((group_r_pli, group_u_pli))
                    y_pli = np.hstack((y_r_pli, y_u_pli))

                    X = np.hstack((X_aec, X_pli))
                    if np.array_equal(y_aec, y_pli):
                        print("Y-values equal")
                        y = y_aec
                    if np.array_equal(group_aec, group_pli):
                        print("group-values equal")
                        group = group_aec

                final_acc_filename = commons.OUTPUT_DIR + f"models/final_SVC_{k}_c_{c}_resp_unres_{s}.pickle"

                #build pipeline with best model
                pipe = Pipeline([
                    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                    ('scaler', StandardScaler()),
                    ('CLF', clf)])

                accuracies, f1s, cms = classify_loso(X, y, group, pipe)

                clf_data = {
                    'accuracies': accuracies,
                    'f1s': f1s,
                    'cms': cms,
                    #'best_params': best_params,
                }

                final_acc_file = open(final_acc_filename, 'ab')
                pickle.dump(clf_data, final_acc_file)
                final_acc_file.close()
                print(sum(accuracies))

                # run first-last
                clf = SVC(max_iter=10000, kernel=k, C=c)

                print(f"Model: Graph {graph} at first_vs_last_step_{s}")
                final_acc_filename = commons.OUTPUT_DIR + f"models/final_SVC_{k}_c_{c}_first_last_{s}.pickle"

                if graph != "both":
                    print(f"MODE {graph}")
                    print(f"FINAL Model Graph {graph} at emf5 vs eml5")
                    X, y, group = filter_dataframe(graph, 'emf5', 'eml5', s)

                if graph == "both":
                    print(f"MODE {graph}")
                    print(f"FINAL Model Graph {graph} at emf5 vs eml5")
                    X_pli, y_pli, group_pli = filter_dataframe('pli', 'emf5', 'eml5', s)
                    X_aec, y_aec, group_aec = filter_dataframe('aec', 'emf5', 'eml5', s)
                    X = np.hstack((X_aec, X_pli))
                    if np.array_equal(y_aec, y_pli):
                        print("Y-values equal")
                        y = y_aec
                    if np.array_equal(group_aec, group_pli):
                        print("group-values equal")
                        group = group_aec

                #build pipeline with best model
                pipe = Pipeline([
                    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                    ('scaler', StandardScaler()),
                    ('CLF', clf)])

                # needed to differentiate the labels for the later pipeline
                # replace one condition to have the label 1 (for the binary comparison)
                # THIS DOES NOT CHANGE THE ACTUAL SELECTED PHASE
                y[y == 3] = 1

                accuracies, f1s, cms = classify_loso(X, y, group, pipe)

                clf_data = {
                    'accuracies': accuracies,
                    'f1s': f1s,
                    'cms': cms,
                    #'best_params': best_params,
                }

                final_acc_file = open(final_acc_filename, 'ab')
                pickle.dump(clf_data, final_acc_file)
                final_acc_file.close()
                print(sum(accuracies))

    for graph in GRAPHS:

        # run resp vs unresp
        final_acc_filename = commons.OUTPUT_DIR + f"models/final_LDA_resp_unres_{s}.pickle"
        clf = LinearDiscriminantAnalysis()

        if graph != "both":
            print(f"MODE {graph}")
            print(f"FINAL Model Graph {graph} at resp_vs_unres")
            # responsive
            X_r, y_r, group_r = filter_dataframe_multiple(graph, 'ec1', 'ec8', 'ind', s)
            y_r[:] = 1
            # unresponsive
            X_u, y_u, group_u = filter_dataframe(graph, 'emf5', 'eml5', s)
            y_u[:] = 4  # randomly choosen integer just need to be different from 1

            # add responsive and unresponsive data together
            X = np.vstack((X_r, X_u))
            group = np.hstack((group_r, group_u))
            y = np.hstack((y_r, y_u))

        if graph == "both":
            print(f"MODE {graph}")
            print(f"FINAL Model Graph {graph} at resp vs unres")
            # responsive
            X_r_aec, y_r_aec, group_r_aec = filter_dataframe_multiple('aec', 'ec1', 'ec8', 'ind', s)
            y_r_aec[:] = 1
            # unresponsive
            X_u_aec, y_u_aec, group_u_aec = filter_dataframe('aec', 'emf5', 'eml5', s)
            y_u_aec[:] = 4  # randomly choosen integer just need to be different from 1

            X_aec = np.vstack((X_r_aec, X_u_aec))
            group_aec = np.hstack((group_r_aec, group_u_aec))
            y_aec = np.hstack((y_r_aec, y_u_aec))

            # responsive
            X_r_pli, y_r_pli, group_r_pli = filter_dataframe_multiple('pli', 'ec1', 'ec8', 'ind', s)
            y_r_pli[:] = 1

            # unresponsive
            X_u_pli, y_u_pli, group_u_pli = filter_dataframe('pli', 'emf5', 'eml5', s)
            y_u_pli[:] = 4  # randomly choosen integer just need to be different from 1

            X_pli = np.vstack((X_r_pli, X_u_pli))
            group_pli = np.hstack((group_r_pli, group_u_pli))
            y_pli = np.hstack((y_r_pli, y_u_pli))

            X = np.hstack((X_aec, X_pli))
            if np.array_equal(y_aec, y_pli):
                print("Y-values equal")
                y = y_aec
            if np.array_equal(group_aec, group_pli):
                print("group-values equal")
                group = group_aec

        #build pipeline with best model
        pipe = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scaler', StandardScaler()),
            ('CLF', clf)])

        accuracies, f1s, cms = classify_loso(X, y, group, pipe)

        clf_data = {
            'accuracies': accuracies,
            'f1s': f1s,
            'cms': cms,
            #'best_params': best_params,
        }

        final_acc_file = open(final_acc_filename, 'ab')
        pickle.dump(clf_data, final_acc_file)
        final_acc_file.close()
        print(sum(accuracies))


        # run first-last
        final_acc_filename = commons.OUTPUT_DIR + f"models/final_LDA_first_last_{s}.pickle"
        clf = LinearDiscriminantAnalysis()

        if graph != "both":
            print(f"MODE {graph}")
            print(f"FINAL Model Graph {graph} at emf5 vs eml5")
            X, y, group = filter_dataframe(graph, 'emf5', 'eml5', s)

        if graph == "both":
            print(f"MODE {graph}")
            print(f"FINAL Model Graph {graph} at emf5 vs eml5")
            X_pli, y_pli, group_pli = filter_dataframe('pli', 'emf5', 'eml5', s)
            X_aec, y_aec, group_aec = filter_dataframe('aec', 'emf5', 'eml5', s)
            X = np.hstack((X_aec, X_pli))
            if np.array_equal(y_aec, y_pli):
                print("Y-values equal")
                y = y_aec
            if np.array_equal(group_aec, group_pli):
                print("group-values equal")
                group = group_aec

        #build pipeline with best model
        pipe = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scaler', StandardScaler()),
            ('CLF', clf)])

        accuracies, f1s, cms = classify_loso(X, y, group, pipe)

        clf_data = {
            'accuracies': accuracies,
            'f1s': f1s,
            'cms': cms,
            #'best_params': best_params,
        }

        final_acc_file = open(final_acc_filename, 'ab')
        pickle.dump(clf_data, final_acc_file)
        final_acc_file.close()
        print(sum(accuracies))


print('The END')