%% Charlotte Maschke November 11 2020
% This script is intended to take the already pre-cut aec and wpli windows
% and generate a csv file containing all the feature required for the
% machine learning analysis.

%% Path Setup

% Remote Source
DATA_PATH = "/home/lotte/projects/def-sblain/lotte/aec_vs_wpli/results/graphs/";
OUTPUT_PATH_1 = "/home/lotte/projects/def-sblain/lotte/aec_vs_wpli/results/features_step01.csv";
OUTPUT_PATH_10 = "/home/lotte/projects/def-sblain/lotte/aec_vs_wpli/results/features_step10.csv";
NEUROALGO_PATH = "/home/lotte/projects/def-sblain/lotte/aec_vs_wpli/NeuroAlgo";


% Add NA library to our path so that we can use it
addpath(genpath(NEUROALGO_PATH));

P_IDS = {'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17'};
EPOCHS = {'eyesclosed_1', 'induction', 'emergence_first', 'emergence_last', 'eyesclosed_8'};
GRAPHS = {'aec','wpli'};
FREQUENCIES = {'alpha'}; % we can only use alpha here since ec8 was only calculated for that

step_size = {1,10}; % in seconds - referring to overlapping and non-overlapping windows
num_regions = 82; % Number of source localized regions


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

% Overwrite the file
delete(OUTPUT_PATH_1);
delete(OUTPUT_PATH_10);

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
for s = 1:length(step_size)
    step = step_size{s};
    for f_i = 1:length(FREQUENCIES)
        frequency = FREQUENCIES{f_i};
        disp(strcat("Frequency: ", frequencyze," Stepsie ", num2str(step)));
        for p_i = 1:length(P_IDS)
            participant = P_IDS(p_i);
            disp(strcat("Participant: ",participant));
            for e_i = 1:length(EPOCHS)
                % Get our variables
                epoch = EPOCHS(e_i);
                disp(strcat("Epochs: ",epoch));

                for g_i = 1:length(GRAPHS)
                   graph = GRAPHS{g_i}; 
                   
                   if step == 1
                   graph_filename = strcat(DATA_PATH,'alpha_step1/',participant,"_",epoch,"_",graph,".mat");
                   elseif step ==10
                   graph_filename = strcat(DATA_PATH,'alpha_step10/',participant,"_",epoch,"_",graph,".mat");
                   end
                   
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

                        % Write both of them into the csv file
                        rows_graph(w_i, :) = [p_i, f_i, e_i, g_i, w_i, mean_graph', std_graph'];
                   end
                   % Writting out to the file the feature calculated
                   if step ==1
                       for w_i = 1:num_window
                       dlmwrite(OUTPUT_PATH_1, rows_graph(w_i, :), '-append');
                       end
                   elseif step == 10
                       for w_i = 1:num_window
                       dlmwrite(OUTPUT_PATH_10, rows_graph(w_i, :), '-append');
                       end
                   end
                end
            end
        end
    end
end

