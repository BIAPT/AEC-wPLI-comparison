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
from commons import permutation_test
from commons import load_pickle, filter_dataframe

# This will be given by the srun in the bash file
# Get the argument
analysis_param = sys.argv[1]

# Parse the parameters
(graph, epoch, steps) = analysis_param.split("_")

print(f"Permutation: Graph {graph} at ec1 vs {epoch} with steps of {steps}")

permutation_filename = commons.OUTPUT_DIR + f"permutation/permutation_Final_model_{graph}_ec1_vs_{epoch}_step_{steps}.csv"
perm_data = pd.DataFrame(np.zeros((1, 5)))
names=['epoch','graph','Random Mean', 'Accuracy', 'p-value']
perm_data.columns=names
c=0

if graph != "both":
    X, y, group = filter_dataframe(graph, epoch, steps)

if graph == "both":
    X_pli, y_pli, group_pli = filter_dataframe('pli', epoch, steps)
    X_aec, y_aec, group_aec = filter_dataframe('aec', epoch, steps)
    X = np.hstack((X_aec, X_pli))
    if np.array_equal(y_aec, y_pli):
        print("Y-values equal")
        y = y_aec
    if np.array_equal(group_aec, group_pli):
        print("group-values equal")
        group = group_aec

clf = commons.best_model

pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

# Training and bootstrap interval generation
acc, perms, p_value = permutation_test(X, y, group, pipe, num_permutation=1000)

# Print out some high level summary
print("Random:")
print(np.mean(perms))
print("Actual Improvement")
print(acc)
print("P Value:")
print(p_value)

perm_data.loc[c, 'epoch'] = epoch
perm_data.loc[c, 'graph'] = graph
perm_data.loc[c, 'Random Mean'] = np.mean(perms)
perm_data.loc[c, 'Accuracy'] = acc
perm_data.loc[c, 'p-value'] = p_value

perm_data.to_csv(permutation_filename, index=False, header= True, sep=',')
print('finished')
