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

# Add the directory containing your module to the Python path (wants absolute paths)
# Here we add to the path everything in the top level
# Since the code will be called from generate_jobs.bash the path that needs to be added is the
# one from the bash script (top level)
scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))

# Common import shared across analysis
import commons
from commons import classify_loso, print_summary
from commons import load_pickle, filter_dataframe, filter_dataframe_multiple
from sklearn.svm import SVC

GRAPHS = ["aec", "pli", "both"]
Steps = ['01', '10']
clf = commons.best_model

# Run deep vs light unconsciousness
for s in Steps:
    for graph in GRAPHS:
        final_acc_filename = commons.OUTPUT_DIR + f"final_models/FINAL_MODEL_{graph}_eml5_vs_emf5_step_{s}.pickle"

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
        print(sum(f1s))

# Run responsiveness vs. unresponsiveness
for s in Steps:
    for graph in GRAPHS:
        final_acc_filename = commons.OUTPUT_DIR + f"final_models/FINAL_MODEL_{graph}_resp_vs_unres_step_{s}.pickle"

        if graph != "both":
            print(f"MODE {graph}")
            print(f"FINAL Model Graph {graph} at resp_vs_unres")
            # responsive
            X_r, y_r, group_r = filter_dataframe_multiple(graph, 'ec1', 'ec8', 'ind', s)
            for a in y_r:
                a = 'resp'
            # unresponsive
            X_u, y_u, group_u = filter_dataframe(graph, 'emf5', 'eml5', s)
            for a in y_u:
                a = 'unre'

            X = np.vstack((X_r, X_u))
            group = np.vstack((group_r, group_u))
            y = np.vstack((y_r, y_u))

        if graph == "both":
            print(f"MODE {graph}")
            print(f"FINAL Model Graph {graph} at resp vs unres")
            # responsive
            X_r_aec, y_r_aec, group_r_aec = filter_dataframe_multiple('aec', 'ec1', 'ec8', 'ind', s)
            for a in y_r_aec:
                a = 'resp'
            # unresponsive
            X_u_aec, y_u_aec, group_u_aec = filter_dataframe('aec', 'emf5', 'eml5', s)
            for a in y_u_aec:
                a = 'unre'
            X_aec = np.vstack((X_r_aec, X_u_aec))
            group_aec = np.vstack((group_r_aec, group_u_aec))
            y_aec = np.vstack((y_r_aec, y_u_aec))

            # responsive
            X_r_pli, y_r_pli, group_r_pli = filter_dataframe_multiple('pli', 'ec1', 'ec8', 'ind', s)
            for a in y_r_pli:
                a = 'resp'
            # unresponsive
            X_u_pli, y_u_pli, group_u_pli = filter_dataframe('pli', 'emf5', 'eml5', s)
            for a in y_u_pli:
                a = 'unre'

            X_pli = np.vstack((X_r_pli, X_u_pli))
            group_pli = np.vstack((group_r_pli, group_u_pli))
            y_pli = np.vstack((y_r_pli, y_u_pli))

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
        print(sum(f1s))


