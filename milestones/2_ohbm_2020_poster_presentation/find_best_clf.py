import sys
import pandas as pd
import numpy as np 

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import LeaveOneGroupOut

from ml_tools.classification import classify_loso_model_selection
import config as cfg


def print_summary(accuracies, group, df):
    """
    Helper function to print a summary of a classifier performance
    :param accuracies: a list of the accuracy obtained across fold (participant)
    :param group: ids of the participants windows
    :param df: dataframe containing all the data about the participant
    :return: None
    """

    p_ids = np.unique(group)
    print("Accuracies: ")
    for accuracy, p_id in zip(accuracies, p_ids):
        print(f"Participant {p_id}: accuracy = {accuracy}")

    print(f"Mean accuracy: {np.mean(accuracies)}")

# Create LOSO Grid Search to search amongst many classifier
class DummyEstimator(BaseEstimator):
    """Dummy estimator to allow gridsearch to test many estimator"""

    def fit(self): pass
    
    def score(self): pass


# Get the argument
analysis_param = sys.argv[1]

best_clf_filename = cfg.OUTPUT_DIR + f'best_clf_{analysis_param}.pickle'

# Parse the parameters
(graph, epoch, feature_group) = analysis_param.split("_")

print(f"Graph {graph} at ec1 vs {epoch} with feature {feature_group}")

# Read the CSV
df = pd.read_csv(cfg.DF_FILE_PATH)

# Keep only the Graph of interest
df = df[df.graph == cfg.GRAPHS.index(graph)]

# Keep only the epoch of interest
df = df[(df.epoch == cfg.EPOCHS[epoch]) | (df.epoch == cfg.EPOCHS['ec1'])]

# Keep only the features of interest
df.drop(df.filter(regex=cfg.FILTER_REGEX[feature_group]), axis=1, inplace=True)

# Set the up the feature matrix, the label vector and the group ids
X = df.drop(['p_id', 'frequency', 'epoch','graph','window'], axis=1).to_numpy()
y = df.epoch.to_numpy()
group = df.p_id.to_numpy()

# Create a pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('clf', DummyEstimator())])  # Placeholder Estimator

# Candidate learning algorithms and their hyperparameters
search_space = [{'clf': [LogisticRegression()],  
                 'clf__penalty': ['l2'],
                 'clf__solver': ['lbfgs'],
                 'clf__max_iter': [1000],
                 'clf__C': np.logspace(0, 4, 10)},

                {'clf': [LinearDiscriminantAnalysis()],
                 'clf__solver': ['svd', 'lsqr']},

                {'clf': [LinearSVC()],
                 'clf__C': [1, 10, 100, 1000]},

                {'clf': [RandomForestClassifier()],
                 'clf__n_estimators': np.linspace(200, 2000, 4, dtype=int),
                 'clf__max_depth': np.linspace(10, 100, 4),
                 'clf__min_samples_split': [3, 10],
                 'clf__min_samples_leaf': [1, 4]},

                {'clf': [DecisionTreeClassifier()],  # Actual Estimator
                 'clf__criterion': ['gini', 'entropy']}]

# We will try to use as many processor as possible for the gridsearch
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
print_summary(accuracies, group, df)