%% Yacine Mahdid June 11 2020
% This script is intended to take the already pre-cut aec and wpli windows
% and generate a csv file containing all the feature required for the
% machine learning analysis.
% 
% participant to different files and merge them afterward)

%% Path Setup
% Local Source
%{
DATA_PATH = "/media/yacine/My Book/test_result/";
OUTPUT_PATH = "/media/yacine/My Book/features.csv";
NUM_CPU = 2;
%}

% Remote Source
%
DATA_PATH = "/lustre03/project/6010672/yacine08/aec_vs_pli/result/graphs/";
OUTPUT_PATH = "/lustre03/project/6010672/yacine08/aec_vs_pli/result/features.csv";
NEUROALGO_PATH = "/lustre03/project/6010672/yacine08/NeuroAlgo";

distcomp.feature( 'LocalUseMpiexec', false ) % This was because of some bug happening in the cluster
% Add NA library to our path so that we can use it
addpath(genpath(NEUROALGO_PATH));

NUM_CPU = 40;
%}

%% Compute Canada Setup

% Create a "local" cluster object
local_cluster = parcluster('local');

% Modify the JobStorageLocation to $SLURM_TMPDIR
pc.JobStorageLocation = strcat('/scratch/yacine08/', getenv('SLURM_JOB_ID'));

% Start the parallel pool
parpool(local_cluster, NUM_CPU)

% we can only use alpha here since ec8 was only calculated for that
P_IDS = {'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17'};
EPOCHS = {'eyesclosed_1', 'emergence_first', 'emergence_last', 'eyesclosed_8'};
GRAPHS = {'aec','wpli'};
FREQUENCIES = {'alpha'};

% Graph theory paramters
num_regions = 82; % Number of source localized regions
num_null_network = 100; % Number of null network to create 
bin_swaps = 10;  % used to create the null network
weight_frequency = 0.1; % used to create the null network
t_level = 0.1; % Threshold level (keep 10%)
transform = 'log'; % this is used for the weighted_global_efficiency

%% Write the header of the CSV file

header = ["p_id", "frequency", "epoch", "graph", "window"];
for r_i = 1:num_regions
   mean_header = strcat("mean_",string(r_i));
   header = [header, mean_header];
end

for r_i = 1:num_regions
    std_header = strcat("std_",string(r_i));
    header = [header,std_header];
end

for r_i = 1:num_regions
    clust_coeff = strcat("wei_clust_coeff_ ", string(r_i));
    header = [header,clust_coeff];      
end

header = [header, "wei_norm_avg_clust_coeff", "wei_norm_g_eff", "wei_community", "wei_small_worldness"];

for r_i = 1:num_regions
    clust_coeff = strcat("bin_clust_coeff_ ", string(r_i));
    header = [header,clust_coeff];      
end

header = [header, "bin_norm_avg_clust_coeff", "bin_norm_g_eff", "bin_community", "bin_small_worldness"];


% Overwrite the file
delete(OUTPUT_PATH);

% Write header to the features file
fileId = fopen(OUTPUT_PATH,'w');
for i = 1:(length(header)-1)
    fprintf(fileId,'%s,',header(i));
end
fprintf(fileId,"%s\n",header(length(header)));
fclose(fileId);

%% Write the body of the CSV file containing the data
% We iterate over all the possible permutation and create our filename to
% load
for f_i = 1:length(FREQUENCIES)
    frequency = FREQUENCIES{f_i};
    disp(strcat("Frequency: ", frequency));
    for p_i = 1:length(P_IDS)
        participant = P_IDS(p_i);
        disp(strcat("Participant: ",participant));
        for e_i = 1:length(EPOCHS)
            % Get our variables
            epoch = EPOCHS(e_i);
            disp(strcat("Epochs: ",epoch));

            for g_i = 1:length(GRAPHS)
               graph = GRAPHS{g_i}; 

               graph_filename = strcat(DATA_PATH,participant,"_",epoch,"_",graph,".mat");
               graph_data = load(graph_filename);

               if strcmp(graph, "aec")
                    graph_data = graph_data.result.aec;              
               else
                    graph_data = graph_data.result.wpli;
               end

               [~,~,num_window] = size(graph_data);
               rows_graph = zeros(num_window, length(header));

               parfor w_i = 1:num_window
                    disp(strcat("Window : ", string(w_i)));
                    single_graph = squeeze(graph_data(:,:,w_i));

                    % Functional connectivity features
                    mean_graph = mean(single_graph,2);
                    std_graph = std(single_graph,0,2);

                    % Weighted Graph Feature
                    X_graph_wei = generate_weighted_graph_feature_vector(single_graph, num_null_network, bin_swaps, weight_frequency, transform);

                    % Binarized Graph Feature
                    X_graph_bin = generate_binary_graph_feature_vector(single_graph, num_null_network, bin_swaps, weight_frequency, t_level);

                    % Write both of them into the csv file
                    rows_graph(w_i, :) = [p_i, f_i, e_i, g_i, w_i, mean_graph', std_graph', X_graph_wei', X_graph_bin'];
                end

                % Writting out to the file the feature calculated
                for w_i = 1:num_window
                    dlmwrite(OUTPUT_PATH, rows_graph(w_i,:), '-append');
                end
            end
        end
    end
end

function [X] = generate_binary_graph_feature_vector(graph, num_null_network, bin_swaps, weight_frequency, t_level)
%GENERATE_FEATURE_VECTOR calculate graph theory feature
%   This is based on experiment_1 and will calculate the following feature
%   vector:
% -> clust_coeff 82x1
% -> norm_avg_clust_coeff 1x1
% -> norm_g_eff 1x1
% -> community 1x1
% -> small_worldness 1x1
% X is a 86x1 feature vector and the first 82 map to the source localized
% regions
%
   
    % Threshold the matrix
    t_grap = threshold_matrix(graph,t_level);
    % Binarize the matrix
    b_graph = binarize_matrix(t_grap);
    % Generate the null networks
    null_networks = generate_null_networks(b_graph, num_null_network, bin_swaps, weight_frequency);

    %% Calculate each of the binary graph theory metric
    % Binary Clustering Coefficient
    [~,norm_g_eff,~,~] = binary_global_efficiency(b_graph,null_networks);

    % Binary Modularity
    community = modularity(b_graph);

    % Binary Smallworldness
    b_small_worldness = undirected_binary_small_worldness(b_graph,null_networks);

    % Binary Clustering Coefficient
    [clust_coeff, norm_avg_clust_coeff] = undirected_binary_clustering_coefficient(b_graph,null_networks);
    
    %% Features vector construction
    X = [clust_coeff; norm_avg_clust_coeff; norm_g_eff;community;b_small_worldness];
end

function [X] = generate_weighted_graph_feature_vector(graph, num_null_network, bin_swaps, weight_frequency, transform)
%GENERATE_FEATURE_VECTOR calculate graph theory feature
%   This is building on the experiment using binary graph classification
%   vector:
% -> mean 82x1
% -> std 82x1
% -> clust_coeff 82x1
% -> norm_avg_clust_coeff 1x1
% -> norm_g_eff 1x1
% -> community 1x1
% -> small_worldness 1x1
% X is a 86x1 feature vector and the first 82 map to the source localized
% regions
%
% graph here is a functional connectivity matrix
    
    % Generate the null networks
    null_networks = generate_null_networks(graph, num_null_network, bin_swaps, weight_frequency);

    %% Calculate each of the weighted graph theory metric
    % Weighted Clustering Coefficient
    % Here we are using the log transform, however I'm not sure if I need
    % to use the inverse distance
    [~,norm_g_eff,~,~] = weighted_global_efficiency(graph, null_networks, transform);

    % Modularity
    community = modularity(graph);

    % Weighted Smallworldness
    w_small_worldness = undirected_weighted_small_worldness(graph,null_networks,transform);

    % Binary Clustering Coefficient
    [clust_coeff, norm_avg_clust_coeff] = undirected_weighted_clustering_coefficient(graph,null_networks);
    
    %% Features vector construction
    X = [clust_coeff; norm_avg_clust_coeff; norm_g_eff; community; w_small_worldness];
end
