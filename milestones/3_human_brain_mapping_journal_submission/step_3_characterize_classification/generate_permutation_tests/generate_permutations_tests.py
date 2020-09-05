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
from commons import permutation_test
from commons import load_pickle, find_best_model, filter_dataframe

# This will be given by the srun in the bash file
# Get the argument
analysis_param = sys.argv[1]

permutation_filename = commons.OUTPUT_DIR + f"permutation_{analysis_param}.pickle"
best_clf_filename = commons.OUTPUT_DIR + f'best_clf_{analysis_param}.pickle'

# Parse the parameters
(graph, epoch, feature_group) = analysis_param.split("_")

print(f"Permutation: Graph {graph} at ec1 vs {epoch} with feature {feature_group}")

X, y, group = filter_dataframe(graph, epoch, feature_group)

best_clf_data = load_pickle(best_clf_filename)
clf = find_best_model(best_clf_data['best_params'])
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('SVM', clf)])

# Training and bootstrap interval generation
acc, perms, p_value = permutation_test(X, y, group, pipe, num_permutation=1000)

# Print out some high level summary
print("Random:")
print(np.mean(perms))
print("Actual Improvement")
print(acc)
print("P Value:")
print(p_value)

# Save the data to disk
perms_file = open(permutation_filename, 'ab')
pickle.dump(perms, perms_file)
perms_file.close()