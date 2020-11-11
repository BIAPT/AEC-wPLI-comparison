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
from commons import load_pickle, find_best_model, filter_dataframe
from sklearn.svm import SVC

# This will be given by the srun in the bash file
# Get the argument
EPOCH = {"emf5","eml5"} # to compare against baseline
GRAPHS = ["aec", "pli"]

DF_FILE_PATH = "/home/lotte/projects/def-sblain/lotte/aec_vs_wpli/results/features.csv";
df = pd.read_csv(DF_FILE_PATH)
features=df.columns[5:]

clf_data=pd.DataFrame(np.zeros((4,166)))
names=['epoch','graph']
names.extend(features)
clf_data.columns=names
c=0
for graph in GRAPHS:
    for epoch in EPOCH:
        clf = SVC(max_iter=100000, kernel='linear', C=0.1)
        clf_data.loc[c,'epoch']=epoch
        clf_data.loc[c,'graph']=graph

        print (f"MODE {graph}")
        print(f"FINAL SVC Model at Graph {graph} at ec1 vs {epoch}")
        X, y, group = filter_dataframe(graph, epoch)

        #build pipeline with best model
        pipe = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scaler', StandardScaler()),
            ('CLF', clf)])

        accuracies, cms = classify_loso(X, y, group, pipe)
        coeff = clf.coef_[0]
        clf_data.loc[c,features] = coeff
        c += 1

clf_data.to_csv("C:/Users/User/Documents/1_MASTER/LAB/AEC_PLI/feature_weights.csv", index=False, header= True, sep=',')
print('finished')

