# General Import
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pickle
import numpy as np
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_pickle(filename):
    """Helper function to unpickle the pickled python obj"""
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()

    return data

pdf = matplotlib.backends.backend_pdf.PdfPages("C:/Users/User/Documents/1_MASTER/LAB/AEC_PLI/AEC_wPLI_all_models.pdf")
all_acc={}

# Path Initiatilization
IN_DIR = "C:/Users/User/Documents/1_MASTER/LAB/AEC_PLI/all_models/"
RESULT_PATH = "C:/Users/User/Documents/1_MASTER/LAB/AEC_PLI/results/"

# This will be given by the srun in the bash file
# Get the argument
EPOCHS = {"emf5","eml5"} # compared against baseline
GRAPHS = ["aec", "pli","both"]

Cs= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
kernels = ['linear']

for c in Cs:
    for k in kernels:
        wPLI_light=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_pli_eml5_raw.pickle")
        wPLI_deep=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_pli_emf5_raw.pickle")
        aec_light=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_aec_eml5_raw.pickle")
        aec_deep=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_aec_emf5_raw.pickle")
        both_light=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_both_eml5_raw.pickle")
        both_deep=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_both_emf5_raw.pickle")

        acc=[np.array(wPLI_light['accuracies'])*100,
             np.array(wPLI_deep['accuracies'])*100,
             np.array(aec_light['accuracies'])*100,
             np.array(aec_deep['accuracies'])*100,
             np.array(both_light['accuracies'])*100,
             np.array(both_deep['accuracies'])*100]

        all_acc[f'SCV_{k}_c_{c}']=np.mean(acc)

        fig = plt.figure()
        sns.boxplot(data=acc).set_xticklabels(['wPLI_light','wPLI_deep','aec_light','aec_deep','both_light','both_deep'])
        plt.title(f'SVC_{k}_c_{c}_mean_acc_{np.mean(acc)}')
        pdf.savefig(fig)
        plt.close(fig)


wPLI_light=load_pickle(f"{IN_DIR}final_LDA_pli_eml5_raw.pickle")
wPLI_deep=load_pickle(f"{IN_DIR}final_LDA_pli_emf5_raw.pickle")
aec_light=load_pickle(f"{IN_DIR}final_LDA_aec_eml5_raw.pickle")
aec_deep=load_pickle(f"{IN_DIR}final_LDA_aec_emf5_raw.pickle")
both_light=load_pickle(f"{IN_DIR}final_LDA_both_eml5_raw.pickle")
both_deep=load_pickle(f"{IN_DIR}final_LDA_both_emf5_raw.pickle")

acc=[np.array(wPLI_light['accuracies'])*100,
     np.array(wPLI_deep['accuracies'])*100,
     np.array(aec_light['accuracies'])*100,
     np.array(aec_deep['accuracies'])*100,
     np.array(both_light['accuracies'])*100,
     np.array(both_deep['accuracies'])*100]

all_acc[f'LDA'] = np.mean(acc)

fig = plt.figure()
sns.boxplot(data=acc).set_xticklabels(['wPLI_light','wPLI_deep','aec_light','aec_deep','both_light','both_deep'])
plt.title(f'LDA_mean_acc_{np.mean(acc)}')
pdf.savefig(fig)
plt.close(fig)

pdf.close()

with open('C:/Users/User/Documents/1_MASTER/LAB/AEC_PLI/all_models_acc.txt', 'w') as f:
    print(all_acc, file=f)
    print(all_acc, file=f)

print('The END')