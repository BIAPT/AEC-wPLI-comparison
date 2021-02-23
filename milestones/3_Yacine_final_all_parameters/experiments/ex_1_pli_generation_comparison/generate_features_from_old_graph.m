%% Yacine Mahdid 8 Septembre 2020
% The goal of this script is to generate the features.csv matrix
% using the old graph data that we can't reproduce in order to find
% out if we can reproduce our result.


%% Compute Canada Setup
NEUROALGO_PATH = "/lustre03/project/6010672/yacine08/NeuroAlgo";

% Add NA library to our path so that we can use it
addpath(genpath(NEUROALGO_PATH));

%% Experimental Variables
DATA_PATH = "/lustre03/project/6010672/yacine08/aec_vs_pli/data/graph_10/";
OUTPUT_PATH = "/lustre03/project/6010672/yacine08/aec_vs_pli/result/features.csv";

% we can only use alpha here since ec8 was only calculated for that
FREQUENCIES = ["alpha"]; 
EPOCHS = ["ec1", "emf5", "eml5", "ec8"];

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
    frequency = FREQUENCIES(f_i);
    disp(strcat("Frequency: ",frequency));
    for e_i = 1:length(EPOCHS)
       % Get our variables
       epoch = EPOCHS(e_i);
       disp(strcat("Epochs: ",epoch));

       % Here we process one file and we need to create the filename
       % Need to process both aec and pli at the same time to equalize them
       aec_filename = strcat(DATA_PATH,"aec_",epoch,"_",frequency,"_aal.mat");
       pli_filename = strcat(DATA_PATH,"pli_",epoch,"_",frequency,"_aal.mat");
       
       % we load it
       aec_data = load(aec_filename);
       aec_data = aec_data.AEC_OUT;
       pli_data = load(pli_filename);
       pli_data = pli_data.PLI_OUT;
       
       num_participant = length(aec_data);
       
       % Iterate on each participants
       for p_i = 1:num_participant
           disp(strcat("Participant id: ", string(p_i)));
            % fix aec reverse orientation compared to pli
            aec_data{p_i} = permute(aec_data{p_i},[3 2 1]);
            
            % match the size of the two datasets
            pli_window_length = size(pli_data{p_i},1);
            aec_window_length = size(aec_data{p_i},1);

            min_window_length = min([pli_window_length aec_window_length]);
            pli_data{p_i} = pli_data{p_i}(1:min_window_length,:,:);
            aec_data{p_i} = aec_data{p_i}(1:min_window_length,:,:);
            
            % calculate the feature for both aec and pli
            rows_aec = zeros(min_window_length, length(header));
            rows_pli = zeros(min_window_length, length(header));
            parfor w_i = 1:min_window_length
                disp(strcat("Window : ", string(w_i)));
                aec_graph = squeeze(aec_data{p_i}(w_i,:,:));
                pli_graph = squeeze(pli_data{p_i}(w_i,:,:));
                
                % Functional connectivity features
                mean_aec = mean(aec_graph,2);
                std_aec = std(aec_graph,0,2);
                
                mean_pli = mean(pli_graph,2);
                std_pli = std(pli_graph,0,2);
                
                % Weighted Graph Feature
                X_aec_wei = generate_weighted_graph_feature_vector(aec_graph, num_null_network, bin_swaps, weight_frequency, transform);
                X_pli_wei = generate_weighted_graph_feature_vector(pli_graph, num_null_network, bin_swaps, weight_frequency, transform);
                
                % Binarized Graph Feature
                X_aec_bin = generate_binary_graph_feature_vector(aec_graph, num_null_network, bin_swaps, weight_frequency, t_level);
                X_pli_bin = generate_binary_graph_feature_vector(pli_graph, num_null_network, bin_swaps, weight_frequency, t_level);
                
                % Write both of them into the csv file
                rows_aec(w_i, :) = [p_i, f_i, e_i, 1, w_i, mean_aec', std_aec', X_aec_wei', X_aec_bin'];
                rows_pli(w_i, :) = [p_i, f_i, e_i, 2, w_i, mean_pli', std_pli', X_pli_wei', X_pli_bin'];
            end
            
            % Writting out to the file the feature calculated
            for w_i = 1:min_window_length
                dlmwrite(OUTPUT_PATH, rows_aec(w_i,:), '-append');
                dlmwrite(OUTPUT_PATH, rows_pli(w_i,:), '-append');
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