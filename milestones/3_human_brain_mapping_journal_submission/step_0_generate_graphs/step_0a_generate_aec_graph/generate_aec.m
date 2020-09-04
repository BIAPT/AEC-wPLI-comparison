%% Yacine Mahdid August 18 2020
% This script goal is to generate the AEC matrices that are needed to
% calculate the features for each participants from the source localized
% data.

%% Path Setup
% Local Source
%{
INPUT_DIR = "/media/yacine/My Book/datasets/consciousness/AEC vs wPLI/source localized data/";
OUTPUT_DIR = "/media/yacine/My Book/test_result/";
NUM_CPU = 2;
%}

% Remote Source
%
INPUT_DIR = "/lustre03/project/6010672/yacine08/aec_vs_pli/data/source_localized_data/";
OUTPUT_DIR = "/lustre03/project/6010672/yacine08/aec_vs_pli/result/graphs/";
NEUROALGO_PATH = "/lustre03/project/6010672/yacine08/NeuroAlgo";

% Add NA library to our path so that we can use it
addpath(genpath(NEUROALGO_PATH));

%}

%% Experiment Variables
P_IDS = {'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17'};
EPOCHS = {'eyesclosed_1', 'induction', 'emergence_first', 'emergence_last', 'eyesclosed_8'};

% indice of the scalp regions
SCALP_REGIONS = [82 62 54 56 58 60 30 26 34 32 28 24 36 86 66 76 84 74 72 70 88 3 78 52 50 48 5 22 46 38 40 98 92 90 96 94 68 16 18 20 44 83 63 55 57 59 61 31 27 35 33 29 25 37 87 67 77 85 75 71 73 89 4 79 53 51 49 6 23 47 39 41 99 93 91 97 95 69 17 19 21 45];
NUM_REGIONS = length(SCALP_REGIONS);

% AEC Parameters:
% Alpha bandpass
low_frequency = 8;
high_frequency = 13;

% Size of the cuts for the data
window_size = 10; % in seconds
step_size = 5; % in seconds

% cuts edge points from hilbert transform
cut = 10;

% Type of graph to calculate
graph = 'aec';

% Participant Path (will need to iterate at the end over all the files)
for p = 1:length(P_IDS)
   p_id = P_IDS{p};
   for e = 1:length(EPOCHS)
        epoch = EPOCHS{e};
        
        fprintf("Analyzing participant '%s' at epoch '%s'\n", p_id, epoch);
        
        participant_in_path = strcat(INPUT_DIR, p_id, filesep, p_id, '_', epoch, '.mat');
        participant_out_path = strcat(OUTPUT_DIR, p_id, '_', epoch, '_', graph, '.mat');
 
        %% Load data
        load(participant_in_path);

        Value = Value(SCALP_REGIONS,:);
        Atlas.Scouts = Atlas.Scouts(SCALP_REGIONS);

        % Get ROI labels from atlas
        LABELS = cell(1,NUM_REGIONS);
        for ii = 1:NUM_REGIONS
            LABELS{ii} = Atlas.Scouts(ii).Label;
        end

        % Sampling frequency : need to round
        fd = 1/(Time(2)-Time(1));

        %% Filtering
        % Frequency filtering, requires eeglab or other frequency filter.
        Vfilt = filter_bandpass(Value, fd, low_frequency, high_frequency);
        Vfilt = Vfilt';

        % number of time points and Regions of Interest
        num_points = length(Vfilt);

        %% Slice up the data into windows
        filtered_data = Vfilt;

        sampling_rate = fd; % in Hz
        [windowed_data, num_window] = create_sliding_window(filtered_data, window_size, step_size, sampling_rate);

        %% Iterate over each window and calculate pairwise corrected aec
        result = struct();
        aec = zeros(NUM_REGIONS, NUM_REGIONS, num_window);

        parfor win_i = 1:num_window
           disp(strcat("AEC at window: ",string(win_i)," of ", string(num_window))); 
           segment_data = squeeze(windowed_data(win_i,:,:));
           aec(:,:, win_i) = aec_pairwise_corrected(segment_data, NUM_REGIONS, cut);
        end

        % Average amplitude correlations over all windows with pairwise
        % correction. Correction is asymmetric so we take the average of the
        % elements above and below the diagonal:
        % e.g. ( corr(env(1)', env(2)) +  corr(env(1),env(2)') )/2,
        % where (1) is an ROI and env' indicates a corrected envelope.
        result.aec = (aec + permute(aec,[2,1,3]))/2; 

        % Bundling some metadata that could be useful along with the graph
        result.window_size = window_size;
        result.step_size = step_size;
        result.labels = LABELS;

        % Save the result structure at the right spot
        save(participant_out_path, 'result');
      
   end
end

% This function is to get overlapping windowed data
function [windowed_data, num_window] = create_sliding_window(data, window_size, step_size, sampling_rate)
%% CREATE SLIDING WINDOW will slice up the data into windows and return them
    %
    % input:
    % data: the points*num regions matrix representing the data
    % window_size: the size of the window in seconds
    % step_size: the size of the step in seconds
    % sampling_rate: the sampling rate of the recording
    %
    % output:
    % windowed_data: the sliced up data which is now a
    % num_window*point*channel tensor
    % num_window: the number of window in the windowed_data
    
    [length_data, num_region] = size(data);
    
    % Need to round from seconds -> points conversion since points are
    % integer valued
    window_size = round(window_size*sampling_rate); % in points
    step_size = round(step_size*sampling_rate); % in points
    
    num_window = length(1:step_size:(length_data - window_size));
    
    windowed_data = zeros(num_window, window_size, num_region);
    index = 1;
    for i = 1:step_size:(length_data - window_size)
        windowed_data(index,:,:) = data(i:i+window_size-1, :);
        index = index + 1;
    end
    
end

function [aec] = aec_pairwise_corrected(data, num_regions, cut)
%% AEC PAIRWISE CORRECTED helper function to calculate the pairwise corrected aec
%
% input:
% data: the data segment to calculate pairwise corrected aec on
% num_regions: number of regions
% cut: the amount we need to remove from the hilbert transform
%
% output:
% aec: a num_region*num_region matrix which has the amplitude envelope
% correlation between two regions
    
    aec = zeros(num_regions, num_regions);
        
    %% Pairwise leakage correction in window for AEC
    % Loops around all possible ROI pairs
    for region_i = 1:num_regions
        y = data(:, region_i);
        for region_j =  1:num_regions
            
            % Skip the correlation between itself
            if region_i == region_j
               continue 
            end
            
            x = data(:, region_j);
            
            % Leakage Reduction
            beta_leak = pinv(y)*x;
            xc = x - y*beta_leak;            
                       
            ht = hilbert([xc,y]);
            ht = ht(cut+1:end-cut,:);
            ht = bsxfun(@minus,ht,mean(ht,1));
            
            % Envelope
            env = abs(ht);
            c = corr(env);
            
            aec(region_i,region_j) = c(1,2);
        end
    end
    
    % Set the diagonal to 0 
    aec(:,:) = aec(:,:).*~eye(num_regions);
end