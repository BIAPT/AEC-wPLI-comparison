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

#import models
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Add the directory containing your module to the Python path (wants absolute paths)
# Here we add to the path everything in the top level
# Since the code will be called from generate_jobs.bash the path that needs to be added is the
# one from the bash script (top level)
scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))

# Common import shared across analysis
import commons
from commons import classify_loso, print_summary
from commons import load_pickle, filter_dataframe

# This will be given by the srun in the bash file
# Get the argument
EPOCHS = {"emf5","eml5"}
GRAPHS = ["aec", "pli"]

Cs= [0.1, 0.4, 0.5, 1, 10]
kernels = ['linear','rbf']
for c in Cs:
    for k in kernels:
        clf=SVC(max_iter=10000, kernel=k, c=c)

        for graph in GRAPHS:
            for epoch in EPOCHS:
                final_acc_filename = commons.OUTPUT_DIR + f"final_SVC_{k}_c_{c}_{graph}_{epoch}_func.pickle"

                print(f"SVC Model {k}_c={c} Graph {graph} at ec1 vs {epoch}")
                X, y, group = filter_dataframe(graph, epoch, 'func')

                #build pipeline with best model
                pipe = Pipeline([
                    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                    ('scaler', StandardScaler()),
                    ('CLF', clf)])

                accuracies, cms = classify_loso(X, y, group, pipe)

                clf_data = {
                    'accuracies': accuracies,
                    #'f1s': f1s,
                    'cms': cms,
                    #'best_params': best_params,
                }

                final_acc_file = open(final_acc_filename, 'ab')
                pickle.dump(clf_data, final_acc_file)
                final_acc_file.close()
                print(sum(accuracies))

clf=LinearDiscriminantAnalysis(max_iter=10000)
for graph in GRAPHS:
    for epoch in EPOCHS:
        final_acc_filename = commons.OUTPUT_DIR + f"final_LDA_{graph}_{epoch}_func.pickle"

        print(f"LDA Model Graph {graph} at ec1 vs {epoch}")
        X, y, group = filter_dataframe(graph, epoch, 'func')

        #build pipeline with best model
        pipe = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scaler', StandardScaler()),
            ('CLF', clf)])

        accuracies, cms = classify_loso(X, y, group, pipe)

        clf_data = {
            'accuracies': accuracies,
            #'f1s': f1s,
            'cms': cms,
            #'best_params': best_params,
        }

        final_acc_file = open(final_acc_filename, 'ab')
        pickle.dump(clf_data, final_acc_file)
        final_acc_file.close()
        print(sum(accuracies))


print('The END')