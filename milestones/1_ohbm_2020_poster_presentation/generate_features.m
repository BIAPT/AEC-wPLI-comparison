%% Yacine Mahdid June 11 2020
% This script is intended to take the already pre-cut aec and wpli windows
% and generate a csv file containing all the feature required for the
% machine learning analysis.
% 
% FIXME: Generate the features from the raw data instead.

% Data are N*82*82 matrices where N is the number of 10 seconds window and
% 82 is the number of channels

% Experimental Variables
DATA_PATH = "/media/yacine/My Book/datasets/consciousness/aec_wpli_source_localized_data/";
FREQUENCY = "alpha"; % we can only use alpha here since ec8 was only calculated for that
TYPES = ["aec", "pli"];
EPOCHS = ["ec1","emf5", "ec8"];

files = dir(DATA_PATH);