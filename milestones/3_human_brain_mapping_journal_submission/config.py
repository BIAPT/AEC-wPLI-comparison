#!/usr/bin/env python

# File and Dir Path Config
DF_FILE_PATH = "/lustre03/project/6010672/yacine08/aec_vs_pli/result/features.csv" # "/home/yacine/Documents/BIAPT/features.csv"
OUTPUT_DIR = "/lustre03/project/6010672/yacine08/aec_vs_pli/result/" #"/home/yacine/Documents/BIAPT/testing/"

EPOCHS = {
    "ec1": 1,
    "emf5": 2,
    "eml5": 3,
    "ec8": 4
}

GRAPHS = ["aec", "pli"]

FILTER_REGEX = {
    "func": "bin|wei",
    "wei": "std|mean|bin",
    "bin": "std|mean|wei",
    "func-wei": "bin",
    "func-bin": "wei"
}
