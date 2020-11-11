# General import
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

# Common import across analysis
import commons
from commons import bootstrap_interval
from commons import load_pickle, find_best_model, filter_dataframe

# This will be given by the srun in the bash file
# Get the argument

# Parse the parameters
EPOCHS = {"ind","emf5","eml5","ec8"} # to compare against baseline
GRAPHS = ["aec", "pli", "both"]

for graph in GRAPHS:
    for epoch in EPOCHS:

        bootstrap_filename = commons.OUTPUT_DIR + f"bootstrap_{graph}_ec1_vs_{epoch}.pickle"
        best_clf_filename = final_acc_filename = commons.OUTPUT_DIR + f"FINAL_MODEL_{graph}_ec1_vs_{epoch}_raw.pickle"

        print(f"Bootstrap: Graph {graph} at ec1 vs {epoch}")

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
        acc_distribution, acc_interval = bootstrap_interval(X, y, group, pipe, num_resample=1000, p_value=0.05)

        # Save the data to disk
        bootstrap_file = open(bootstrap_filename, 'ab')
        bootstrap_data = {
            'distribution': acc_distribution,
            'interval': acc_interval
        }
        pickle.dump(bootstrap_data, bootstrap_file)
        bootstrap_file.close()

        # Print out some high level summary
        print("Accuracy Distribution:")
        print(acc_distribution)
        print(f"Mean: {np.mean(acc_distribution)} and std: {np.std(acc_distribution)}")
        print("Bootstrap Interval")
        print(acc_interval)