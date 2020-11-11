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

pdf = matplotlib.backends.backend_pdf.PdfPages("C:/Users/User/Documents/1_MASTER/LAB/AEC_PLI/AEC_wPLI_FINAL_model.pdf")
all_acc={}

# Path Initiatilization
IN_DIR = "C:/Users/User/Documents/1_MASTER/LAB/AEC_PLI/permutation_results/"
RESULT_PATH = "C:/Users/User/Documents/1_MASTER/LAB/AEC_PLI/results/"

# This will be given by the srun in the bash file
# Get the argument
EPOCHS = {"ind","emf5","eml5","ec8"} # compared against baseline
GRAPHS = ["aec", "pli","both"]

for graph in GRAPHS:
    ind=load_pickle(f"{IN_DIR}permutation_Final_model_{graph}_ec1_vs_ind.pickle")
    emf5=load_pickle(f"{IN_DIR}FINAL_MODEL_{graph}_ec1_vs_emf5_raw.pickle")
    eml5=load_pickle(f"{IN_DIR}FINAL_MODEL_{graph}_ec1_vs_eml5_raw.pickle")
    ec8=load_pickle(f"{IN_DIR}FINAL_MODEL_{graph}_ec1_vs_ec8_raw.pickle")

    acc=[np.array(ind['accuracies'])*100,
         np.array(emf5['accuracies'])*100,
         np.array(eml5['accuracies'])*100,
         np.array(ec8['accuracies'])*100]

    df = pd.DataFrame(data=np.transpose(acc))
    df.columns=['induction', 'deep sedation', 'pre ROC', 'Recovery']
    df.to_csv(f"C:/Users/User/Documents/1_MASTER/LAB/AEC_PLI/FINAL_output_{graph}.txt", index=False)

    fig = plt.figure()
    sns.boxplot(data=acc).set_xticklabels(['induction', 'deep sedation', 'pre ROC', 'Recovery'])
    plt.title(f'FINAL_SVC_model_{graph}')
    pdf.savefig(fig)
    plt.close(fig)

pdf.close()

print('The END')