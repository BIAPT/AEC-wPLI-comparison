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

# This will be given by the srun in the bash file
# Get the argument
GRAPHS = ["aec", "pli","both"]
Steps = ['01', '10']

for s in Steps:
    pdf = matplotlib.backends.backend_pdf.PdfPages(commons.OUTPUT_DIR + f"NEW_AEC_wPLI_FINAL_model_accuracies_step_{s}.pdf")

    # Path Initiatilization
    IN_DIR = commons.OUTPUT_DIR
    RESULT_PATH = commons.OUTPUT_DIR


    for graph in GRAPHS:
        ind=load_pickle(f"{IN_DIR}final_models/FINAL_MODEL_{graph}_ec1_vs_ind_step_{s}.pickle")
        emf5=load_pickle(f"{IN_DIR}final_models/FINAL_MODEL_{graph}_ec1_vs_emf5_step_{s}.pickle")
        eml5=load_pickle(f"{IN_DIR}final_models/FINAL_MODEL_{graph}_ec1_vs_eml5_step_{s}.pickle")
        ec8=load_pickle(f"{IN_DIR}final_models/FINAL_MODEL_{graph}_ec1_vs_ec8_step_{s}.pickle")
        last_first =load_pickle(f"{IN_DIR}final_models/FINAL_MODEL_{graph}_eml5_vs_emf5_step_{s}.pickle")
        resp_unres =load_pickle(f"{IN_DIR}final_models/FINAL_MODEL_{graph}_resp_vs_unres_step_{s}.pickle")

        acc=[np.array(ind['accuracies'])*100,
             np.array(emf5['accuracies'])*100,
             np.array(eml5['accuracies'])*100,
             np.array(ec8['accuracies'])*100,
             np.array(last_first['accuracies']) * 100,
             np.array(resp_unres['accuracies']) * 100
             ]

        f1=[np.array(ind['f1s'])*100,
            np.array(emf5['f1s'])*100,
            np.array(eml5['f1s'])*100,
            np.array(ec8['f1s'])*100,
            np.array(last_first['f1s']) * 100,
            np.array(resp_unres['f1s']) * 100
            ]

        df = pd.DataFrame(data=np.transpose(acc))
        df.columns=['induction', 'deep sedation', 'pre ROC', 'Recovery','last_first', 'resp_unres']
        df.to_csv(f"{commons.OUTPUT_DIR}FINAL_ACC_{graph}_step{s}.csv", index=False)

        df = pd.DataFrame(data=np.transpose(f1))
        df.columns=['induction', 'deep sedation', 'pre ROC', 'Recovery','last_first', 'resp_unres']
        df.to_csv(f"{commons.OUTPUT_DIR}FINAL_F1_{graph}_step{s}.csv", index=False)

        fig = plt.figure()
        sns.boxplot(data=acc).set_xticklabels(['induction', 'deep sedation', 'pre ROC',
                                               'Recovery','last_first', 'resp_unres'])
        plt.title(f'FINAL_SVC_model_ACC_{graph}')
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()

print('The END')