"""
Charlotte Maschke November 11 2020
This script will combine all files with accuracies and models, which can be summarized and visualized in a pdf file.
"""
# General Import
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pickle
import pandas as pd
import numpy as np
import seaborn as sns

# Add the directory containing your module to the Python path (wants absolute paths)
scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))

import commons
from commons import load_pickle

EPOCHS = {"emf5", "eml5"}  # compared against baseline
GRAPHS = ["aec", "pli", "both"]

Cs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
kernels = ['linear']
Steps = ['01', '10']

for s in Steps:
    pdf = matplotlib.backends.backend_pdf.PdfPages(commons.OUTPUT_DIR + f"models/AEC_wPLI_all_extra_models_step_{s}.pdf")
    all_acc={}
    all_f1s={}

    # Path Initiatilization
    IN_DIR = commons.OUTPUT_DIR + "models/"

    for c in Cs:
        for k in kernels:
            first_last_pli = load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_pli_first_last_{s}.pickle")
            first_last_aec = load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_aec_first_last_{s}.pickle")
            first_last_both = load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_both_first_last_{s}.pickle")
            resp_unres_pli = load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_pli_resp_unres_{s}.pickle")
            resp_unres_aec = load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_aec_resp_unres_{s}.pickle")
            resp_unres_both = load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_both_resp_unres_{s}.pickle")

            acc=[np.array(first_last_pli['accuracies'])*100,
                 np.array(first_last_aec['accuracies'])*100,
                 np.array(first_last_both['accuracies'])*100,
                 np.array(resp_unres_pli['accuracies'])*100,
                 np.array(resp_unres_aec['accuracies'])*100,
                 np.array(resp_unres_both['accuracies'])*100]

            f1s=[np.array(first_last_pli['f1s']),
                 np.array(first_last_aec['f1s']),
                 np.array(first_last_both['f1s']),
                 np.array(resp_unres_aec['f1s']),
                 np.array(resp_unres_pli['f1s']),
                 np.array(resp_unres_both['f1s'])]

            all_acc[f'SCV_{k}_c_{c}']=np.mean(acc)
            all_f1s[f'SCV_{k}_c_{c}']=np.mean(f1s)

            fig = plt.figure()
            sns.boxplot(data=acc).set_xticklabels(['states_pli','states_aec','states_both','resp_pli','resp_aec','resp_both'])
            plt.title(f'SVC_{k}_c_{c}_acc_{np.mean(acc)}_f1_{np.mean(f1s)}')
            pdf.savefig(fig)
            plt.close(fig)

    first_last_pli = load_pickle(f"{IN_DIR}final_LDA_pli_first_last_{s}.pickle")
    first_last_aec = load_pickle(f"{IN_DIR}final_LDA_aec_first_last_{s}.pickle")
    first_last_both = load_pickle(f"{IN_DIR}final_LDA_both_first_last_{s}.pickle")
    resp_unres_pli = load_pickle(f"{IN_DIR}final_LDA_pli_resp_unres_{s}.pickle")
    resp_unres_aec = load_pickle(f"{IN_DIR}final_LDA_aec_resp_unres_{s}.pickle")
    resp_unres_both = load_pickle(f"{IN_DIR}final_LDA_both_resp_unres_{s}.pickle")

    acc = [np.array(first_last_pli['accuracies']) * 100,
           np.array(first_last_aec['accuracies']) * 100,
           np.array(first_last_both['accuracies']) * 100,
           np.array(resp_unres_pli['accuracies']) * 100,
           np.array(resp_unres_aec['accuracies']) * 100,
           np.array(resp_unres_both['accuracies']) * 100]

    f1s = [np.array(first_last_pli['f1s']),
           np.array(first_last_aec['f1s']),
           np.array(first_last_both['f1s']),
           np.array(resp_unres_aec['f1s']),
           np.array(resp_unres_pli['f1s']),
           np.array(resp_unres_both['f1s'])]

    all_acc[f'LDA'] = np.mean(acc)
    all_f1s[f'LDA'] = np.mean(f1s)

    fig = plt.figure()
    sns.boxplot(data=acc).set_xticklabels(['states_pli','states_aec','states_both','resp_pli','resp_aec','resp_both'])
    plt.title(f'LDA_mean_acc_{np.mean(acc)}_f1_{np.mean(f1s)}')
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()

    with open(commons.OUTPUT_DIR + f"all_extra_models_step_{s}_acc.txt", 'w') as f:
        print(all_acc, file=f)
        print(all_f1s, file=f)
        print(max(all_acc, key=all_acc.get), file=f)
        print(max(all_f1s, key=all_f1s.get), file=f)

print('The END')