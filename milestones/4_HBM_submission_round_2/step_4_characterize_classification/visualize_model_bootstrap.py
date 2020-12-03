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
from operator import add
from operator import sub

# Add the directory containing your module to the Python path (wants absolute paths)
#scriptpath = "."
#sys.path.append(os.path.abspath(scriptpath))

# This will be given by the srun in the bash file
# Get the argument
EPOCHS = {"ind","emf5","eml5","ec8"} # compared against baseline
GRAPHS = ["aec", "pli","both"]
Steps = ['01', '10']

def load_pickle(filename):
    '''Helper function to unpickle the pickled python obj'''
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()

    return data

for s in Steps:
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"AEC_wPLI_FINAL_model_bootstrap_step_{s}.pdf")

    IN_DIR = f"C:/Users/User/Desktop/AEC_wPLI_FINAL_19_11_2020/step4_characterize_model/bootstrap/step{s}/"

    for graph in GRAPHS:
        ind=pd.read_csv(f"{IN_DIR}bootstrap_Final_model_{graph}_ec1_vs_ind_step_{s}.csv")
        emf5=pd.read_csv(f"{IN_DIR}bootstrap_Final_model_{graph}_ec1_vs_emf5_step_{s}.csv")
        eml5=pd.read_csv(f"{IN_DIR}bootstrap_Final_model_{graph}_ec1_vs_eml5_step_{s}.csv")
        ec8=pd.read_csv(f"{IN_DIR}bootstrap_Final_model_{graph}_ec1_vs_ec8_step_{s}.csv")

        acc_mean=[np.array(ind['Acc_Dist Mean'][0])*100,
             np.array(emf5['Acc_Dist Mean'][0])*100,
             np.array(eml5['Acc_Dist Mean'][0])*100,
             np.array(ec8['Acc_Dist Mean'][0])*100]

        acc_sd=[np.array(ind['Acc_Dist Std'][0])*100,
             np.array(emf5['Acc_Dist Std'][0])*100,
             np.array(eml5['Acc_Dist Std'][0])*100,
             np.array(ec8['Acc_Dist Std'][0])*100]

        min=[np.array(ind['acc_interval_low'][0])*100,
             np.array(emf5['acc_interval_low'][0])*100,
             np.array(eml5['acc_interval_low'][0])*100,
             np.array(ec8['acc_interval_low'][0])*100]

        max=[np.array(ind['acc_interval_high'][0])*100,
             np.array(emf5['acc_interval_high'][0])*100,
             np.array(eml5['acc_interval_high'][0])*100,
             np.array(ec8['acc_interval_high'][0])*100]

        fig = plt.figure()
        plt.bar([0,1,2,3], list(map(add, acc_mean, acc_sd)), color='C0')
        plt.bar([0,1,2,3], acc_mean, color='C1')
        plt.bar([0,1,2,3], np.array(acc_mean)-0.5, color='C0')
        plt.bar([0,1,2,3], list(map(sub, acc_mean, acc_sd)), color='white')
        plt.errorbar((0,0),(min[0],max[0]), color='black')
        plt.errorbar((1,1),(min[1],max[1]), color='black')
        plt.errorbar((2,2),(min[2],max[2]), color='black')
        plt.errorbar((3,3),(min[3],max[3]), color='black')
        plt.xticks([0,1,2,3], ['induction', 'deep sedation', 'pre ROC', 'Recovery'])
        plt.title(f'FINAL_SVC_model_ACC_{graph}')
        plt.ylim(40,100)
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()

print('The END')