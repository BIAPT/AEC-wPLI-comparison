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
import numpy as np
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import commons
from commons import load_pickle

EPOCHS = {"emf5", "eml5"}  # compared against baseline
GRAPHS = ["aec", "pli", "both"]

Cs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
kernels = ['linear']
Steps = ['01', '10']

for s in Steps:
    pdf = matplotlib.backends.backend_pdf.PdfPages(commons.OUTPUT_DIR + f"models/AEC_wPLI_all_models_step_{s}.pdf")
    all_acc={}
    all_f1s={}

    # Path Initiatilization
    IN_DIR = commons.OUTPUT_DIR + "models/"

    for c in Cs:
        for k in kernels:
            wPLI_light=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_pli_eml5_{s}.pickle")
            wPLI_deep=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_pli_emf5_{s}.pickle")
            aec_light=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_aec_eml5_{s}.pickle")
            aec_deep=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_aec_emf5_{s}.pickle")
            both_light=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_both_eml5_{s}.pickle")
            both_deep=load_pickle(f"{IN_DIR}final_SVC_{k}_c_{c}_both_emf5_{s}.pickle")

            acc=[np.array(wPLI_light['accuracies'])*100,
                 np.array(wPLI_deep['accuracies'])*100,
                 np.array(aec_light['accuracies'])*100,
                 np.array(aec_deep['accuracies'])*100,
                 np.array(both_light['accuracies'])*100,
                 np.array(both_deep['accuracies'])*100]

            f1s=[np.array(wPLI_light['f1s']),
                 np.array(wPLI_deep['f1s']),
                 np.array(aec_light['f1s']),
                 np.array(aec_deep['f1s']),
                 np.array(both_light['f1s']),
                 np.array(both_deep['f1s'])]

            all_acc[f'SCV_{k}_c_{c}']=np.mean(acc)
            all_f1s[f'SCV_{k}_c_{c}']=np.mean(f1s)

            fig = plt.figure()
            sns.boxplot(data=acc).set_xticklabels(['wPLI_light','wPLI_deep','aec_light','aec_deep','both_light','both_deep'])
            plt.title(f'SVC_{k}_c_{c}_acc_{np.mean(acc)}_f1_{np.mean(f1s)}')
            pdf.savefig(fig)
            plt.close(fig)


    wPLI_light=load_pickle(f"{IN_DIR}final_LDA_pli_eml5_{s}.pickle")
    wPLI_deep=load_pickle(f"{IN_DIR}final_LDA_pli_emf5_{s}.pickle")
    aec_light=load_pickle(f"{IN_DIR}final_LDA_aec_eml5_{s}.pickle")
    aec_deep=load_pickle(f"{IN_DIR}final_LDA_aec_emf5_{s}.pickle")
    both_light=load_pickle(f"{IN_DIR}final_LDA_both_eml5_{s}.pickle")
    both_deep=load_pickle(f"{IN_DIR}final_LDA_both_emf5_{s}.pickle")

    acc=[np.array(wPLI_light['accuracies'])*100,
         np.array(wPLI_deep['accuracies'])*100,
         np.array(aec_light['accuracies'])*100,
         np.array(aec_deep['accuracies'])*100,
         np.array(both_light['accuracies'])*100,
         np.array(both_deep['accuracies'])*100]

    f1s = [np.array(wPLI_light['f1s']),
           np.array(wPLI_deep['f1s']),
           np.array(aec_light['f1s']),
           np.array(aec_deep['f1s']),
           np.array(both_light['f1s']),
           np.array(both_deep['f1s'])]

    all_acc[f'LDA'] = np.mean(acc)
    all_f1s[f'LDA'] = np.mean(f1s)

    fig = plt.figure()
    sns.boxplot(data=acc).set_xticklabels(['wPLI_light','wPLI_deep','aec_light','aec_deep','both_light','both_deep'])
    plt.title(f'LDA_mean_acc_{np.mean(acc)}_f1_{np.mean(f1s)}')
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()

    with open(commons.OUTPUT_DIR + f"all_models_step_{s}_acc.txt", 'w') as f:
        print(all_acc, file=f)
        print(all_f1s, file=f)
        print(max(all_acc, key=all_acc.get), file=f)
        print(max(all_f1s, key=all_f1s.get), file=f)

print('The END')