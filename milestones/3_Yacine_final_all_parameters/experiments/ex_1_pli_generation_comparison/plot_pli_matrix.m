%% Yacine Mahdid September 16 2020
% This script is used to compare the faulty PLI graphs that were generated
% by Jason with the one with the correct definition but that (during the
% time of writing) is buggy.

%% Global Experiment Variables

% paths
graph_path = "/home/yacine/Documents/BIAPT/AEC vs wPLI/data/ex_1_pli_generation_comparison/pli_surrogate_graphs/";
output_path = "/home/yacine/Documents/BIAPT/AEC vs wPLI/result/ex_1_pli_generation_comparison/pli_surrogate_graphs/";

% Variables
P_IDS = {'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17'};
EPOCHS = {'eyesclosed_1', 'induction', 'emergence_first', 'emergence_last', 'eyesclosed_8'};
GRAPHS = {'wpli'};
NUM_REGIONS = 82;

% Labels for the plots
region_table = readtable('regions_ordering.csv');
labels = region_table.label;

%% Setup the Directory Structure
mkdir(output_path)
mkdir(strcat(output_path,"/matrix/"))

%% Connectivity Matrices for Average Participant
fixed_rois = get_fixed_rois();
for g_i = 1:length(GRAPHS)
    graph = GRAPHS{g_i};

    avg_participant_graph = zeros(NUM_REGIONS, NUM_REGIONS, length(EPOCHS));
    for e_i = 1:length(EPOCHS)
        epoch = EPOCHS{e_i};

        avg_graphs = zeros(NUM_REGIONS, NUM_REGIONS, length(P_IDS));
        for p_i = 1:length(P_IDS)
            participant = P_IDS{p_i};
            
            disp(strcat("At ", graph, " ", epoch, " ", participant));

            % Load the data
            filename = strcat(graph_path, participant, "_", epoch, "_", graph, ".mat");
            graph_data = load(filename);

            if strcmp(graph, "aec")
                functional_connectivity = graph_data.result.aec;
            else
                functional_connectivity = graph_data.result.wpli;
            end

            % Average over all windows
            avg_graphs(:,:,p_i) = mean(functional_connectivity, 3);
        end

        % Average all the participant
        avg_participant_graph(:,:,e_i) = mean(avg_graphs, 3);
    end
    
 
    % Gathering the min/max threshold for the figures
    mean_connectivity = mean(avg_participant_graph(:));
    std_connectivity = std(avg_participant_graph(:));

    % Mean
    avg_avg_participant_graph = squeeze(mean(avg_participant_graph,2));
    all_mean_weights = avg_avg_participant_graph(:);
    
    % Std
    avg_std_participant_graph = squeeze(std(avg_participant_graph,0,2));
    all_std_weights = avg_std_participant_graph(:);

    map = jet(length(all_mean_weights));
    fprintf("Min: %.2f and Max: %.2f for Mean at %s\n", min(all_mean_weights), max(all_mean_weights), graph);
    fprintf("Min: %.2f and Max: %.2f for Std at %s\n", min(all_std_weights), max(all_std_weights), graph);
    
    % Generating the figures
    for e_i = 1:length(EPOCHS)
        epoch = EPOCHS{e_i};
        
        %% Generate the figure for connectivity matrix
        output_filename = strcat(output_path, "matrix/", epoch, "_", graph, ".png");
        title_name = strcat("Average Participant ", graph, " at ", epoch);
        
        avg_graph = avg_participant_graph(:,:,e_i);
        
        % Generate the connectivity matrix and save it at the right spot
        generate_connectivity(avg_graph, mean_connectivity, ...,
            std_connectivity, labels, output_filename, title_name);
         
    end   
    

end


function generate_connectivity(connectivity_matrix, mean, std, labels, filename, title_name)
%% PLOT CONNECTIVITY helper function to plot the connectivity matrix and save generate the connectivity matrix
%   This helper function will make the matrix in the right way for
%   generating the final figure and then save it at the right spot
%
%   input
%   connectivity_matrix: a N*N matrix representing the connectivity
%   mean & std: this is the mean and standard deviation
%   labels: the labels to put on the x and y axis
%   filename: the output filename
%   title_name: the title name to put on the top of matrix figure

    figure;
    imagesc(connectivity_matrix);
    caxis([mean - 2*std, mean + 2*std]);
    xticklabels(labels);
    xtickangle(90);
    xticks(1:length(labels));
    yticks(1:length(labels));
    yticklabels(labels);
    title(title_name);
    colormap('jet');
    colorbar;
    set(gcf, 'Position',  [100, 100, 800, 600])

    saveas(gcf, filename)
    close(gcf);
end

