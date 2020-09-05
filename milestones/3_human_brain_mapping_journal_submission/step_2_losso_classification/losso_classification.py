# General import
import os
import sys

# Data science import
import pickle
import numpy as np 

# Sklearn import
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Add the directory containing your module to the Python path (wants absolute paths)
# Here we add to the path everything in the top level
scriptpath = "../" 
sys.path.append(os.path.abspath(scriptpath))

# Import common functionalities
import commons
from commons import load_pickle, find_best_model
from commons import classify_loso_model_selection
from commons import DummyEstimator, print_summary, filter_dataframe, search_space

# Get the argument
analysis_param = sys.argv[1]

best_clf_filename = commons.OUTPUT_DIR + f'best_clf_{analysis_param}.pickle'

# Parse the parameters
(graph, epoch, feature_group) = analysis_param.split("_")

print(f"Graph {graph} at ec1 vs {epoch} with feature {feature_group}")

X,y,group = filter_dataframe(graph, epoch, feature_group)

# Create a pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('clf', DummyEstimator())])  # Placeholder Estimator

# We will use as many processor as possible for the gridsearch
gs = GridSearchCV(pipe, search_space, cv=LeaveOneGroupOut(), n_jobs=-1)

accuracies, f1s, cms, best_params = classify_loso_model_selection(X, y, group, gs)

# Saving the performance metrics
clf_data = {
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
    'best_params': best_params,
}

best_clf_file = open(best_clf_filename, 'ab')
pickle.dump(clf_data, best_clf_file)
best_clf_file.close()

# Print out the summary in the console
print_summary(accuracies, group)
