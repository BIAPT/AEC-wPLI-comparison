%% Yacine Mahdid August 24 2020
% The purpose of this script is to generate the following three type of
% connectivity figure:
% - Connectivity matrices figure for the average participant across the whole 5 minutes
% - mean global connectivity on the brain for the average participant for the whole 5 minutes
% - std of the global connectivity on the brain for the average participant for the whole 5 minutes
%
% The above three figure needs to be repeated for AEC and wPLI

%% Global Experiment Variables

% paths
graph_path = "/home/yacine/Documents/BIAPT/graphs/";
output_path = "/home/yacine/Documents/BIAPT/aec_vs_pli_result/";

% Variables
P_IDS = {'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17'};
EPOCHS = {'eyesclosed_1', 'induction', 'emergence_first', 'emergence_last', 'eyesclosed_8'};
GRAPHS = {'aec', 'wpli'};
NUM_REGIONS = 82;
MAP = colormap('jet');

% Labels for the plots
region_table = readtable('regions_ordering.csv');
labels = region_table.label;

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
    
    
    % Generating the figures
    for e_i = 1:length(EPOCHS)
        epoch = EPOCHS{e_i};
        
        % Generate the figure for connectivity matrix
        output_filename = strcat(output_path, epoch, "_", graph, ".png");
        
        figure;
        imagesc(avg_participant_graph(:,:,e_i));
        caxis([mean_connectivity - 2*std_connectivity, mean_connectivity + 2*std_connectivity]);
        xticklabels(labels);
        xtickangle(90);
        xticks(1:length(labels));
        yticks(1:length(labels));
        yticklabels(labels);
        title(strcat("Average Participant ", graph, " at ", epoch));
        colormap('jet');
        colorbar;
        set(gcf, 'Position',  [100, 100, 800, 600])
        
        saveas(gcf, output_filename)
        close(gcf);
        
    end
    
    
    % Generate the figure for connectivity matrix
    %{

    % Figure for mean connectivity
    output_filename = strcat(output_path, epoch, "_", graph, "_mean.png");
    make_figure_top(fixed_rois, MAP, mean(avg_participant_graph,2))
    title(strcat("Average Participant Mean Connectivity ", graph, " at ", epoch));
    saveas(gcf, output_filename)
    %close(gcf);

    % Figure for std connectivity
    output_filename = strcat(output_path, epoch, "_", graph, "_std.png");
    make_figure_top(fixed_rois, MAP, std(avg_participant_graph,0,2))
    title(strcat("Average Participant Std Connectivity  ", graph, " at ", epoch));
    saveas(gcf, output_filename)
    %close(gcf);
    %}
    
    

end