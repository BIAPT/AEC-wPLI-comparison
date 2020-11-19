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
from commons import load_pickle, filter_dataframe
from sklearn.svm import SVC

# This will be given by the srun in the bash file
# Get the argument
EPOCHS = ["ind", "emf5", "eml5", "ec8"]  # compared against baseline
GRAPHS = ["aec", "pli", "both"]
Steps = ['01', '10']

for step in Steps:
    DF_FILE_PATH = commons.OUTPUT_DIR +f"features_step{step}.csv";
    df = pd.read_csv(DF_FILE_PATH)
    features=df.columns[5:]
    clf_data=pd.DataFrame(np.zeros((len(EPOCHS)*len(GRAPHS)+4,167))) #82 regions mean and sd +2
    names=['epoch', 'graph', 'feature']
    names.extend(features)
    clf_data.columns=names
    c=0

    for graph in GRAPHS:
        for epoch in EPOCHS:
            if graph != "both":
                print(f"MODE {graph}")
                print(f"FINAL Model Graph {graph} at ec1 vs {epoch}")
                X, y, group = filter_dataframe(graph, epoch, step)

            if graph == "both":
                print(f"MODE {graph}")
                print(f"FINAL Model Graph {graph} at ec1 vs {epoch}")
                X_pli, y_pli, group_pli = filter_dataframe('pli', epoch, step)
                X_aec, y_aec, group_aec = filter_dataframe('aec', epoch, step)
                X = np.hstack((X_aec, X_pli))
                if np.array_equal(y_aec, y_pli):
                    print("Y-values equal")
                    y = y_aec
                if np.array_equal(group_aec, group_pli):
                    print("group-values equal")
                    group = group_aec

            clf = commons.best_model
            clf_data.loc[c,'epoch']=epoch
            clf_data.loc[c,'graph']=graph

            print (f"MODE {graph}")
            print(f"FINAL SVC Model at Graph {graph} at ec1 vs {epoch}_step_{step}")

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
                 c += 2

    clf_data.to_csv(commons.OUTPUT_DIR +f"feature_weights_{step}.csv", index=False, header= True, sep=',')
print('finished')

