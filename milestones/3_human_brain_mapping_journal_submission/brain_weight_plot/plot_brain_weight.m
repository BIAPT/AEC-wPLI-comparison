%% Yacine Mahdid July 3 2020
% This script is used to plot the weights from the trained ML classifier
% onto the AAL brain atlas.
% 

%% Experimental Variable
% AEC is in shades of hot red-orange
aec_hex = ['#fef0d9';'#fdd49e';'#fdbb84';'#fc8d59';'#ef6548';'#d7301f'; '#990000'];
% PLI is in shades of cold blue
pli_hex = ['#f1eef6'; '#d0d1e6'; '#a6bddb'; '#74a9cf'; '#3690c0'; '#0570b0'; '#034e7b'];

%% Load the data used for plotting
% we will fix the naming space as it is all over the place with the
% saved data.
data = load('ROI_MNI_V5_nii.mat'); 
aal_reg = data.AALreg; % AALreg = volume image of 120 ROIs
data = load('ROI_MNI_V5_vol.mat'); 
rois = data.ROI1; % ROI1 = contains info on ROIs
data = load('Atlas_scouts.mat');  
scouts = data.scout; % scout = info of edited atlas


% data from the table
% 1 == 'emf5' + 'aec'
% 2 == 'eml5' + 'aec'
% 3 == 'emf5' + 'wpli'
% 4 == 'eml5' + 'wpli'
table = readtable('weights_data.csv');
mean_weight = table{1, 3:81+3};
std_weight = table{1, 82+3:end};
%% Set the scouts name and labels
% The scouts are at most of length 82 since we are using only the 
% scalp regions. However the AAL atlas contains 120 regions.
% Therefore we need to get the index of only the regions defined by the
% scouts
num_locs = length(scouts);

% Matches scout ROIs to original AAL atlas
selected_region_index = zeros(1,num_locs);

for i = 1:num_locs
    % Format the labels properly
    label = scouts(i).Label;
    label(end-1)='_';
    
    % Search for a match in the rois
    for j = 1:length(rois)
        if strcmp(rois(j).Nom_L, label)
            selected_region_index(i) = j;
        end
    end
end

%% Fix the ROI
% Here we need to fix the rois to contains the AAL information + the
% information about the right regions.
% HERE NOT REALLY SURE WHAT IS THE DIMENSION OF THE AAL REPRESENTING.
fixed_rois = zeros([size(aal_reg), num_locs]);

for reg = 1:num_locs
    fixed_rois(:,:,:,reg) = aal_reg == rois(selected_region_index(reg)).ID;
end

%% Plot the weights on the brain
% Create the figure

% To update the map we just need to change the Hex here by giving it 7
% values, the interpl will then interpolate for the size required here 82
map = get_color_map(pli_hex);

%mean_weight
make_figure_top(fixed_rois, map, mean_weight)
make_figure_sagittal(fixed_rois, map, mean_weight);

%std_weight
make_figure_top(fixed_rois, map, std_weight);
make_figure_sagittal(fixed_rois, map, std_weight);

function [map] = get_color_map(hex)
    vec = [ 0; 15; 30; 44; 68; 83 ; 100];
    raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
    N = 82;
    map = interp1(vec,raw,linspace(0,100,N),'pchip');
end

function make_figure(fixed_rois, map, weights, region_range, view_range)
    figure
    axis ([0,100,0,50,0,100])

    color = colormap(map);
    sorted_weights = sort(weights);
    hold all

    for reg = region_range
        roisurf=isosurface(fixed_rois(:,:,:,reg),0.5);
        h = trisurf(roisurf.faces,roisurf.vertices(:,1),roisurf.vertices(:,2),roisurf.vertices(:,3));

        weight = weights(reg);
        color_index = find(sorted_weights == weight);
        % THIS IS WHERE THE ALPHA WILL CHANGE DEPENDING ON THE VALUES OF THE W
        set(h,'facecolor',color(color_index,:),'LineWidth',0.1,'LineStyle','none');
    end

    set(gca,'view',view_range)
    axis equal
    axis off
    set(gcf,'color','white')
end

function make_figure_top(fixed_rois, map, weights)
    make_figure(fixed_rois, map, weights, 1:82, [-90 90]);
end

function make_figure_sagittal(fixed_rois, map, weights)
    make_figure(fixed_rois, map, weights, 1:41, [0 0]);
end