#!/usr/bin/env python

# File and Dir Path Config
OUTPUT_DIR = "/home/lotte/projects/def-sblain/lotte/aec_vs_wpli/results/";
DF_FILE_PATH = "/home/lotte/projects/def-sblain/lotte/aec_vs_wpli/results/features.csv";

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
