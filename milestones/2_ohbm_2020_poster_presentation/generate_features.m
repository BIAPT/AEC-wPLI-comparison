%% Yacine Mahdid June 11 2020
% This script is intended to take the already pre-cut aec and wpli windows
% and generate a csv file containing all the feature required for the
% machine learning analysis.
% 
% FIXME: Generate the features from the raw data instead.

% Data are N*82*82 matrices where N is the number of 10 seconds window and
% 82 is the number of channels. (aec however is 82*82*N which is a bit
% cumbersome, it is addressed by permutig the dimension in the data loader)

%% Experimental Variables
DATA_PATH = "/media/yacine/My Book/datasets/consciousness/aec_wpli_source_localized_data/";
FREQUENCY = "alpha"; % we can only use alpha here since ec8 was only calculated for that
TYPES = ["aec", "pli"];
EPOCHS = ["ec1","emf5", "ec8"];

%% Generate each participant feature
files = dir(DATA_PATH);
for id = 3:length(files)
    file = files(id);
    [freq, type, epoch, data] = load_data(file);

    % Skip the data that are not related to our analysis
    if ~strcmp(freq, FREQUENCY) || ~any(strcmp(EPOCHS,epoch))
       continue 
    end
    
    % At this point we need to generate the features
    fprintf("freq = %s, type = %s epoch = %s\n", freq, type, epoch);
    
    


    
end

function [freq, type, epoch, data] = load_data(file)
% Load Data: Will load data from the file and extract metainfo from filename    

    data_path = strcat(file.folder, filesep, file.name);
    
    % Get meta-info
    content  = strsplit(file.name,'_');
    type = content{1};
    epoch = content{2};
    freq = content{3};
    
    raw = load(data_path);
    
    % The field within the raw datastructure depends on the type
    if strcmp(type, "aec")
        data = raw.AEC_OUT;
        
        % We need to permute the dimension of aec to fit pli
        data = permute(data,[3 2 1]);
    else
        data = raw.PLI_OUT;
    end
    
end