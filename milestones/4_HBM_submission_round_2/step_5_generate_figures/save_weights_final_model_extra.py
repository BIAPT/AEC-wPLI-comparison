# General Import
import os
import sys

# Data science import
import pickle
import numpy as np
import pandas as pd
import csv

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

# This will be given by the srun in the bash file
# Get the argument
GRAPHS = ["aec", "pli", "both"]
Steps = ['01', '10']

for step in Steps:
    DF_FILE_PATH = commons.OUTPUT_DIR +f"features_step{step}.csv";
    df = pd.read_csv(DF_FILE_PATH)
    features = df.columns[5:]
    clf_data = pd.DataFrame(np.zeros((2*len(GRAPHS)+2,167))) #82 regions mean and sd +2
    names = ['epoch', 'graph', 'feature']
    names.extend(features)
    clf_data.columns = names
    c = 0

    for graph in GRAPHS:
        if graph != "both":
            print(f"MODE {graph}")
            print(f"FINAL Model Graph {graph} at emf5 vs eml5")
            X, y, group = filter_dataframe(graph, 'emf5', 'eml5', step)

        if graph == "both":
            print(f"MODE {graph}")
            print(f"FINAL Model Graph {graph} at emf5 vs eml5")
            X_pli, y_pli, group_pli = filter_dataframe('pli', 'emf5', 'eml5', step)
            X_aec, y_aec, group_aec = filter_dataframe('aec', 'emf5', 'eml5', step)
            X = np.hstack((X_aec, X_pli))
            if np.array_equal(y_aec, y_pli):
                print("Y-values equal")
                y = y_aec
            if np.array_equal(group_aec, group_pli):
                print("group-values equal")
                group = group_aec

        # needed to differentiate the labels for the later pipeline
        # replace one condition to have the label 1 (for the binary comparison)
        # THIS DOES NOT CHANGE THE ACTUAL SELECTED PHASE
        y[y == 3] = 1

        clf = commons.best_model
        clf_data.loc[c,'epoch']='deep-light'
        clf_data.loc[c,'graph']=graph

        print (f"MODE {graph}")
        print(f"FINAL SVC Model at Graph {graph} at deep-light_step_{step}")

        #build pipeline with best model
        pipe = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scaler', StandardScaler()),
            ('CLF', clf)])

        accuracies, f1, cms = classify_loso(X, y, group, pipe)

        if graph != "both":
            coeff = clf.coef_[0]
            clf_data.loc[c,features] = coeff
            clf_data.loc[c, 'feature'] = graph
            c += 1

        if graph == "both":
             coeff = clf.coef_[0]
             clf_data.loc[c,features] = coeff[:164]
             clf_data.loc[c+1,features] = coeff[164:]
             clf_data.loc[c, 'feature'] = "aec"
             clf_data.loc[c+1, 'feature'] = "pli"
             clf_data.loc[c+1, 'epoch'] = 'deep-light'
             clf_data.loc[c+1, 'graph'] = graph
             c += 2

    for graph in GRAPHS:
        if graph != "both":
            print(f"MODE {graph}")
            print(f"FINAL Model Graph {graph} at resp_vs_unres")
            # responsive
            X_r, y_r, group_r = filter_dataframe_multiple(graph, 'ec1', 'ec8', 'ind', step)
            y_r[:] = 1
            # unresponsive
            X_u, y_u, group_u = filter_dataframe(graph, 'emf5', 'eml5', step)
            y_u[:] = 4  # randomly choosen integer just need to be different from 1

            # add responsive and unresponsive data together
            X = np.vstack((X_r, X_u))
            group = np.hstack((group_r, group_u))
            y = np.hstack((y_r, y_u))

        if graph == "both":
            print(f"MODE {graph}")
            print(f"FINAL Model Graph {graph} at resp vs unres")
            # responsive
            X_r_aec, y_r_aec, group_r_aec = filter_dataframe_multiple('aec', 'ec1', 'ec8', 'ind', step)
            y_r_aec[:] = 1
            # unresponsive
            X_u_aec, y_u_aec, group_u_aec = filter_dataframe('aec', 'emf5', 'eml5', step)
            y_u_aec[:] = 4  # randomly choosen integer just need to be different from 1

            X_aec = np.vstack((X_r_aec, X_u_aec))
            group_aec = np.hstack((group_r_aec, group_u_aec))
            y_aec = np.hstack((y_r_aec, y_u_aec))

            # responsive
            X_r_pli, y_r_pli, group_r_pli = filter_dataframe_multiple('pli', 'ec1', 'ec8', 'ind', step)
            y_r_pli[:] = 1

            # unresponsive
            X_u_pli, y_u_pli, group_u_pli = filter_dataframe('pli', 'emf5', 'eml5', step)
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

        clf = commons.best_model
        clf_data.loc[c,'epoch']='resp_unre'
        clf_data.loc[c,'graph']=graph

        print (f"MODE {graph}")
        print(f"FINAL SVC Model at Graph {graph} at resp_unre_step_{step}")

        #build pipeline with best model
        pipe = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scaler', StandardScaler()),
            ('CLF', clf)])

        accuracies, f1, cms = classify_loso(X, y, group, pipe)

        if graph != "both":
            coeff = clf.coef_[0]
            clf_data.loc[c,features] = coeff
            clf_data.loc[c, 'feature'] = graph
            c += 1

        if graph == "both":
             coeff = clf.coef_[0]
             clf_data.loc[c,features] = coeff[:164]
             clf_data.loc[c+1,features] = coeff[164:]
             clf_data.loc[c, 'feature'] = "aec"
             clf_data.loc[c+1, 'feature'] = "pli"
             clf_data.loc[c+1, 'epoch'] = 'resp_unre'
             clf_data.loc[c+1, 'graph'] = graph
             c += 2

    clf_data.to_csv(commons.OUTPUT_DIR +f"feature_weights_extra_{step}.csv", index=False, header= True, sep=',')
print('finished')

