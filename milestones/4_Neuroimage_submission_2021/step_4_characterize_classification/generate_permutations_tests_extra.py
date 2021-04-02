# General import
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

# Common import across analysis
# Common import shared across analysis
import commons
from commons import permutation_test
from commons import load_pickle, filter_dataframe, filter_dataframe_multiple
from sklearn.svm import SVC

# This will be given by the srun in the bash file
# Get the argument
analysis_param = sys.argv[1]

# Parse the parameters
(graph, s) = analysis_param.split("_")

clf = commons.best_model

"""
# Run deep vs light unconsciousness
print(f"Permutation: Graph {graph} at eml5_vs_emf5_step_{s}")
permutation_filename = commons.OUTPUT_DIR + f"permutation/permutation_10000_Final_model_{graph}_eml5_vs_emf5_step_{s}.csv"
perm_data = pd.DataFrame(np.zeros((1, 6)))
names = ['epoch', 'graph', 'Acc_Dist Mean', 'Acc_Dist Std', 'acc_interval_low', 'acc_interval_high']
perm_data.columns = names
c = 0

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

# Training and permutation test
acc, perms, p_value = permutation_test(X, y, group, pipe, num_permutation=10000)

# Print out some high level summary
print("Random:")
print(np.mean(perms))
print("Actual Improvement")
print(acc)
print("P Value:")
print(p_value)

perm_data.loc[c, 'epoch'] = 'deep-light'
perm_data.loc[c, 'graph'] = graph
perm_data.loc[c, 'Random Mean'] = np.mean(perms)
perm_data.loc[c, 'Accuracy'] = acc
perm_data.loc[c, 'p-value'] = p_value

perm_data.to_csv(permutation_filename, index=False, header=True, sep=',')
print('finished')
"""

print(f"Permutation: Graph {graph} at resp_vs_unre_step_{s}")
permutation_filename = commons.OUTPUT_DIR + f"permutation/permutation_10000_Final_model_{graph}_resp_vs_unre_step_{s}.csv"
perm_data = pd.DataFrame(np.zeros((1, 6)))
names = ['epoch', 'graph', 'Acc_Dist Mean', 'Acc_Dist Std', 'acc_interval_low', 'acc_interval_high']
perm_data.columns = names
c = 0

if graph != "both":
    print(f"MODE {graph}")
    print(f"FINAL Model Graph {graph} at resp_vs_unres")
    # responsive
    X_r, y_r, group_r = filter_dataframe_multiple(graph, 'ec1', 'ec8', 'ind', s)
    y_r[:] = 1
    # unresponsive
    X_u, y_u, group_u = filter_dataframe(graph, 'emf5', 'eml5', s)
    y_u[:] = 4    #randomly choosen integer just need to be different from 1

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
    y_u_aec[:] = 4 #randomly choosen integer just need to be different from 1

    X_aec = np.vstack((X_r_aec, X_u_aec))
    group_aec = np.hstack((group_r_aec, group_u_aec))
    y_aec = np.hstack((y_r_aec, y_u_aec))

    # responsive
    X_r_pli, y_r_pli, group_r_pli = filter_dataframe_multiple('pli', 'ec1', 'ec8', 'ind', s)
    y_r_pli[:] = 1

    # unresponsive
    X_u_pli, y_u_pli, group_u_pli = filter_dataframe('pli', 'emf5', 'eml5', s)
    y_u_pli[:] = 4  #randomly choosen integer just need to be different from 1

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

# Training and permutation test
acc, perms, p_value = permutation_test(X, y, group, pipe, num_permutation=10000)

# Print out some high level summary
print("Random:")
print(np.mean(perms))
print("Actual Improvement")
print(acc)
print("P Value:")
print(p_value)

perm_data.loc[c, 'epoch'] = 'resp_unre'
perm_data.loc[c, 'graph'] = graph
perm_data.loc[c, 'Random Mean'] = np.mean(perms)
perm_data.loc[c, 'Accuracy'] = acc
perm_data.loc[c, 'p-value'] = p_value

perm_data.to_csv(permutation_filename, index=False, header=True, sep=',')

print('finished')

