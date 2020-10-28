%% Variables initialization
% Weights
%data = load('weights.mat');
%weights_pli_pre_ROC = data.weights_pli_pre_ROC;
%weights_aec_pre_ROC = data.weights_aec_pre_ROC;
%weights_pli_unconscious = data.weights_pli_unconscious;
%weights_aec_unconscious = data.weights_aec_unconscious;

% Colormap [nothing = 1, mean = 2, std = 3, both= 4]
color_map = [ 1 1 1 ; 0.4660, 0.6740, 0.1880 ; 0.9290, 0.6940, 0.1250; 0, 0.4470, 0.7410]; % This is the colormap for the not important, mean, std, both

%% Weights vector initialization and USER INPUT
weights = repelem(1,82); %To change using the values above
% Side of brain to map (left or right)

side = 'right';
if(strcmp(side,'left'))
   roi_range = 1:41;
elseif(strcmp(side,'right'))
    roi_range = 42:82;
end

%% Load the necessary variables
load('ROI_MNI_V5_nii.mat'); % AALreg = volume image of 120 ROIs
load('ROI_MNI_V5_vol.mat'); % ROI1 = contains info on ROIs
load('Atlas_scouts.mat'); % scout = info of edited atlas 

%% Set the scouts name and labels
scout = scout(roi_range);

number_locs = length(scout); % number_locs =  82

% Matches scout ROIs to original AAL atlas
labellist = zeros(1,number_locs);

for jj = 1:length(scout)
        ll = scout(jj).Label;
        ll(end-1)='_';
    for ii = 1:length(ROI1)
        if strcmp(ROI1(ii).Nom_L,ll)
            labellist(jj) = ii;
        end
    end
end

%% Fix the ROI
roi = zeros([size(AALreg),number_locs]);
for reg = 1:number_locs
    roi(:,:,:,reg) = AALreg == ROI1(labellist(reg)).ID;
end

%% Plot the brain with the correct weights
% Can specify colour limits, here 0 to 1
aal_brain(roi_range,weights,roi,color_map);
