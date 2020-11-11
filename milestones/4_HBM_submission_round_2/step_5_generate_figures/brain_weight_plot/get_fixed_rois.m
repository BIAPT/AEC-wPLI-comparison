function [fixed_rois] = get_fixed_rois()
    % we will fix the naming space as it is all over the place with the
    % saved data.
    data = load('ROI_MNI_V5_nii.mat'); 
    aal_reg = data.AALreg; % AALreg = volume image of 120 ROIs
    data = load('ROI_MNI_V5_vol.mat'); 
    rois = data.ROI1; % ROI1 = contains info on ROIs
    data = load('Atlas_scouts.mat');  
    scouts = data.scout; % scout = info of edited atlas

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

end

