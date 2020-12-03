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
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"AEC_wPLI_FINAL_model_permutation_step_{s}.pdf")

    IN_DIR = f"C:/Users/User/Desktop/AEC_wPLI_FINAL_19_11_2020/step4_characterize_model/permutation/step{s}/"

    for graph in GRAPHS:
        ind=pd.read_csv(f"{IN_DIR}permutation_Final_model_{graph}_ec1_vs_ind_step_{s}.csv")
        emf5=pd.read_csv(f"{IN_DIR}permutation_Final_model_{graph}_ec1_vs_emf5_step_{s}.csv")
        eml5=pd.read_csv(f"{IN_DIR}permutation_Final_model_{graph}_ec1_vs_eml5_step_{s}.csv")
        ec8=pd.read_csv(f"{IN_DIR}permutation_Final_model_{graph}_ec1_vs_ec8_step_{s}.csv")

        rand=[np.array(ind['Random Mean'][0])*100,
             np.array(emf5['Random Mean'][0])*100,
             np.array(eml5['Random Mean'][0])*100,
             np.array(ec8['Random Mean'][0])*100]

        acc=[np.array(ind['Accuracy'][0])*100,
             np.array(emf5['Accuracy'][0])*100,
             np.array(eml5['Accuracy'][0])*100,
             np.array(ec8['Accuracy'][0])*100]

        p=[str(ind['p-value'][0])[0:4],
             str(emf5['p-value'][0])[0:4],
             str(eml5['p-value'][0])[0:4],
             str(ec8['p-value'][0])[0:4]]

        fig = plt.figure()
        plt.bar([0,1,2,3], acc)
        plt.bar([0,1,2,3], rand)
        plt.xticks([0,1,2,3], ['induction', 'deep sedation', 'pre ROC', 'Recovery'])
        plt.text(-0.2,47,p[0])
        plt.text(0.8,47,p[1])
        plt.text(1.8,47,p[2])
        plt.text(2.8,47,p[3])
        plt.title(f'FINAL_SVC_model_ACC_{graph}')
        plt.ylim(45,90)
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()

print('The END')