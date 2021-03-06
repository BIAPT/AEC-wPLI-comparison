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
graph_path = "/home/yacine/Documents/BIAPT/AEC vs wPLI/data/step_size_5_sec/graphs/";
output_path = "/home/yacine/Documents/BIAPT/AEC vs wPLI/result/step_size_5_seconds/";

% Variables
P_IDS = {'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17'};
EPOCHS = {'eyesclosed_1', 'induction', 'emergence_first', 'emergence_last', 'eyesclosed_8'};
GRAPHS = {'aec', 'wpli'};
NUM_REGIONS = 82;

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
        
        %% Figure for mean connectivity
        cur_mean_weights = mean(avg_graph,2);
        
        % Top
        output_filename = strcat(output_path, "mean/",epoch, "_", graph, "_top_mean.png");
        title_name = strcat("Average Participant Mean Connectivity Top ", graph, " at ", epoch);
        
        plot_brain_top(fixed_rois, map, cur_mean_weights, all_mean_weights);
 
        title(title_name);
        saveas(gcf, output_filename)
        close(gcf);
        
        % sagittal
        output_filename = strcat(output_path, "mean/",epoch, "_", graph, "_sagittal_mean.png");
        title_name = strcat("Average Participant Mean Connectivity Sagittal ", graph, " at ", epoch);
        
        plot_brain_sagittal(fixed_rois, map, cur_mean_weights, all_mean_weights);
 
        title(title_name);
        saveas(gcf, output_filename)
        close(gcf);
        
        %% Figure for std connectivity
        cur_std_weights = std(avg_graph,0,2);
        % Top
        output_filename = strcat(output_path, "std/", epoch, "_", graph, "_top_std.png");
        title_name = strcat("Average Participant Std Connectivity Top ", graph, " at ", epoch);
        
        plot_brain_top(fixed_rois, map, cur_std_weights, all_std_weights);
 
        title(title_name);
        saveas(gcf, output_filename)
        close(gcf);
        
        %sagittal
        output_filename = strcat(output_path, "std/", epoch, "_", graph, "_sagittal_std.png");
        title_name = strcat("Average Participant Std Connectivity Sagittal ", graph, " at ", epoch);
        
        plot_brain_sagittal(fixed_rois, map, cur_std_weights, all_std_weights);
 
        title(title_name);
        saveas(gcf, output_filename)
        close(gcf);    
    end
    
    
    % Generate the figure for connectivity matrix
    %{



    % Figure for std connectivity
    output_filename = strcat(output_path, epoch, "_", graph, "_std.png");
    make_figure_top(fixed_rois, MAP, std(avg_participant_graph,0,2))
    title(strcat("Average Participant Std Connectivity  ", graph, " at ", epoch));
    saveas(gcf, output_filename)
    %close(gcf);
    %}
    
    

end

function plot_brain(fixed_rois, map, cur_weights, all_weights, region_range, view_coordinate)
%% PLOT BRAIN 3D visualization of the brain at a particular angle
% this function will plot the weights onto the brain at a specific starting
% angle which is useful to make 2D views
%
% input
% fixed_rois: these are the region of interest that are calibrated for the
% scalp regions
% map: the colormap to use for the region coloring
% cur_weights: the weights we need to plot
% all_weights: all the weights to base the coloring upon, need to contain
% cur_weights inside to work
% regions_range: the range in index of the regions to show, this help to
% show half a brain for 2D viz
% view_coordinate: the coordinate to start the 3D viz useful to start the
% 2D viz
%
% Note
% We could return the figure handle so that we don't have to call gcf
% everytime we want to plot/save the thing
    
    figure;
    axis ([0,100,0,50,0,100]);

    % Color is a N*3 color where N is of the same size as 
    color = colormap(map);
    sorted_weights = sort(all_weights);
    
    % Plot each regions one by one
    hold all
    for reg = region_range
        roisurf=isosurface(fixed_rois(:,:,:,reg),0.5);
        h = trisurf(roisurf.faces,roisurf.vertices(:,1),roisurf.vertices(:,2),roisurf.vertices(:,3));

        weight = cur_weights(reg);
        color_index = find(sorted_weights == weight);
        
        set(h,'facecolor',color(color_index,:),'LineWidth',0.1,'LineStyle','none');
    end

    set(gca,'view',view_coordinate)
    axis equal
    axis off
    set(gcf,'color','white')
end

function plot_brain_top(fixed_rois, map, cur_weights, all_weights)
%% PLOT BRAIN TOP this is a wrapper function to facilitate the usage of plot_brain
% the goal is to plot the brain from a birdeye view
% 
% input
% fixed_rois: these are the region of interest that are calibrated for the
% scalp regions
% map: the colormap to use for the region coloring
% cur_weights: the weights we need to plot
% all_weights: all the weights to base the coloring upon, need to contain
% cur_weights inside to work

    plot_brain(fixed_rois, map, cur_weights, all_weights, 1:82, [-90 90]);
end

function plot_brain_sagittal(fixed_rois, map, cur_weights, all_weights)
%% PLOT BRAIN TOP this is a wrapper function to facilitate the usage of plot_brain
% the goal is to plot the brain from a sagittal view
% 
% input
% fixed_rois: these are the region of interest that are calibrated for the
% scalp regions
% map: the colormap to use for the region coloring
% cur_weights: the weights we need to plot
% all_weights: all the weights to base the coloring upon, need to contain
% cur_weights inside to work

    plot_brain(fixed_rois, map, cur_weights, all_weights, 1:41, [0 0]);
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

