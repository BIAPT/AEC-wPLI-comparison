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
from commons import classify_loso
from commons import load_pickle, find_best_model, filter_dataframe

# This will be given by the srun in the bash file
# Get the argument
analysis_param = sys.argv[1]

final_acc_filename = commons.OUTPUT_DIR + f"final_acc_{analysis_param}.pickle"
best_clf_filename = commons.OUTPUT_DIR + f'best_clf_{analysis_param}.pickle'

# Parse the parameters
(graph, epoch, feature_group) = analysis_param.split("_")

print(f"FINAL Model: Graph {graph} at ec1 vs {epoch} with feature {feature_group}")

X, y, group = filter_dataframe(graph, epoch, feature_group)

best_clf_data = load_pickle(best_clf_filename)
clf = find_best_model(best_clf_data['best_params'])

#build pipeline with best model
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

accuracies, cms = classify_loso(X, y, group, pipe)

print_summary(accuracies, group, clf.parameters)