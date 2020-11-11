% Yacine Mahdid July 28 2020
% The purpose of this script is to automatically generate all the figures
% we need for the brain visualization of the feature importance (mean and
% standard deviation in the case when this was written). This will
% automatically generate the figures in the right format and the right
% color.

%% Experimental Variable

OUTPUT_DIR = '/home/yacine/Desktop/result_figure';

% AEC is in shades of hot red-orange
aec_hex = ['#fef0d9';'#fdd49e';'#fdbb84';'#fc8d59';'#ef6548';'#d7301f'; '#990000'];
aec_map = get_color_map(aec_hex);
% PLI is in shades of cold blue
pli_hex = ['#f1eef6'; '#d0d1e6'; '#a6bddb'; '#74a9cf'; '#3690c0'; '#0570b0'; '#034e7b'];
pli_map = get_color_map(pli_hex);

% Datastructure we need
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
labels = {'aec_emf5', 'aec_eml5', 'emf5_wpli', 'eml5_wpli'};

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

function make_figure_top(output_name, fixed_rois, map, weights)
    make_figure(output_name, fixed_rois, map, weights, 1:82, [-90 90]);
end

function make_figure_sagittal(output_name, fixed_rois, map, weights)
    make_figure(output_name, fixed_rois, map, weights, 1:41, [0 0]);
end


%% Figure Generation
for i = 1:length(labels)
    label = labels{i};
    display(strcat("Generating: ",label));
    mean_weight = table{i, 3:81+3};
    std_weight = table{i, 82+3:end};
    
    % Get the right colormap
    if (i == 1 || i == 2)
        map = aec_map;
    else
        map = pli_map;
    end
    
    %mean_weight figure both top and side
    output_name = strcat(OUTPUT_DIR, filesep, label, '_mean_top.png');
    make_figure_top(output_name, fixed_rois, map, mean_weight)
    output_name = strcat(OUTPUT_DIR, filesep, label, '_mean_side.png');
    make_figure_sagittal(output_name, fixed_rois, map, mean_weight);

    %std_weight figure both top and side
    output_name = strcat(OUTPUT_DIR, filesep, label, '_std_top.png');
    make_figure_top(output_name, fixed_rois, map, std_weight);
    output_name = strcat(OUTPUT_DIR, filesep, label, '_std_side.png');
    make_figure_sagittal(output_name, fixed_rois, map, std_weight);
end


%% Helper Functions
function [map] = get_color_map(hex)
% GET COLOR MAP helper function to transform hex into colormap
%   input:
%   hex: is a 7 cell cell-array which contains hex code
    vec = [ 0; 15; 30; 44; 68; 83 ; 100];
    raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
    N = 82;
    map = interp1(vec,raw,linspace(0,100,N),'pchip');
end

function make_figure(output_name, fixed_rois, map, weights, region_range, view_range)
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
    
    saveas(gcf,output_name)
    delete(gcf)
end

