%% Yacine Mahdid July 3 2020
% This script is used to plot the weights from the trained ML classifier
% onto the AAL brain atlas.
% 

%% Experimental Variable
% AEC is in shades of hot red-orange
aec_hex = ['#fef0d9';'#fdd49e';'#fdbb84';'#fc8d59';'#ef6548';'#d7301f'; '#990000'];
% PLI is in shades of cold blue
pli_hex = ['#f1eef6'; '#d0d1e6'; '#a6bddb'; '#74a9cf'; '#3690c0'; '#0570b0'; '#034e7b'];

% data from the table
% 1 == 'emf5' + 'aec'
% 2 == 'eml5' + 'aec'
% 3 == 'emf5' + 'wpli'
% 4 == 'eml5' + 'wpli'
table = readtable('weights_data.csv');
mean_weight = table{1, 3:81+3};
std_weight = table{1, 82+3:end};

%% Plot the weights on the brain
% Create the figure
fixed_rois = get_fixed_rois();

% To update the map we just need to change the Hex here by giving it 7
% values, the interpl will then interpolate for the size required here 82
map = get_color_map(pli_hex);

%mean_weight
make_figure_top(fixed_rois, map, mean_weight)
make_figure_sagittal(fixed_rois, map, mean_weight);

%std_weight
make_figure_top(fixed_rois, map, std_weight);
make_figure_sagittal(fixed_rois, map, std_weight);


