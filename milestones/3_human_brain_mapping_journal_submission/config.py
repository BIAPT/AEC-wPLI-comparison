#!/usr/bin/env python

# File and Dir Path Config
DF_FILE_PATH = "/home/yacine/Documents/BIAPT/features.csv" # "/lustre03/project/6010672/yacine08/aec_vs_pli/result/features.csv"
OUTPUT_DIR = "/lustre03/project/6010672/yacine08/aec_vs_pli/result/" #"/home/yacine/Documents/BIAPT/testing/"

EPOCHS = {
    "ec1": 1,
    "if5": 2,
    "emf5": 3,
    "eml5": 4,
    "ec8": 5
}

GRAPHS = ["aec", "pli"]

FILTER_REGEX = {
    "func": "bin|wei",
    "wei": "std|mean|bin",
    "bin": "std|mean|wei",
    "func-wei": "bin",
    "func-bin": "wei"
}
