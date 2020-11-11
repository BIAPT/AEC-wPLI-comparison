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
from commons import load_pickle, find_best_model, filter_dataframe

# Parse the parameters
EPOCHS = {"ind","emf5","eml5","ec8"} # to compare against baseline
GRAPHS = ["aec", "pli", "both"]

permutation_filename = commons.OUTPUT_DIR + f"permutation_Final_model_SUMMARY.csv"
perm_data=pd.DataFrame(np.zeros((len(EPOCHS)*len(GRAPHS),5)))
names=['epoch','graph','Random Mean', 'Accuracy', 'p-value']
perm_data.columns=names
c=0


for graph in GRAPHS:
    for epoch in EPOCHS:

        best_clf_filename = final_acc_filename = commons.OUTPUT_DIR + f"FINAL_MODEL_{graph}_ec1_vs_{epoch}_raw.pickle"

        print(f"Permutation: Graph {graph} at ec1 vs {epoch}")

        if graph != "both":
            print(f"MODE {graph}")
            print(f"FINAL SVC Model at Graph {graph} at ec1 vs {epoch}")
            X, y, group = filter_dataframe(graph, epoch)

        if graph == "both":
            print(f"MODE {graph}")
            print(f"FINAL SVC Model at Graph {graph} at ec1 vs {epoch}")
            X_pli, y_pli, group_pli = filter_dataframe('pli', epoch)
            X_aec, y_aec, group_aec = filter_dataframe('aec', epoch)
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

        c += 1

        # Save the data to disk
        #perms_file = open(permutation_filename, 'ab')
        #pickle.dump([perms, acc, p_value], perms_file)
        #perms_file.close()

perm_data.to_csv(permutation_filename, index=False, header= True, sep=',')
print('finished')
