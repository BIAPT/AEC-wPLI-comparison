import sys
import pickle
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import config as cfg
from ml_tools.classification import bootstrap_interval
from utils import load_pickle, find_best_model, filter_dataframe

# This will be given by the srun in the bash file
# Get the argument
analysis_param = sys.argv[1]

bootstrap_filename = cfg.OUTPUT_DIR + f"bootstrap_{analysis_param}.pickle"
best_clf_filename = cfg.OUTPUT_DIR + f'best_clf_{analysis_param}.pickle'

# Parse the parameters
(graph, epoch, feature_group) = analysis_param.split("_")

print(f"Bootstrap: Graph {graph} at ec1 vs {epoch} with feature {feature_group}")

X, y, group = filter_dataframe(graph, epoch, feature_group)

best_clf_data = load_pickle(best_clf_filename)
clf = find_best_model(best_clf_data['best_params'])
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('SVM', clf)])

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
print("F1 Distribution:")
print(acc_distribution)
print(f"Mean: {np.mean(acc_distribution)} and std: {np.std(acc_distribution)}")
print("Bootstrap Interval")
print(acc_interval)