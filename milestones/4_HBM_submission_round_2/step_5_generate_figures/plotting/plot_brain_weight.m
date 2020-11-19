%% Yacine Mahdid July 3 2020
% This script is used to plot the weights from the trained ML classifier
% onto the AAL brain atlas.
% 

%% Experimental Variable
% AEC is in shades of hot red-orange
aec_hex = ['#fef0d9';'#fdd49e';'#fdbb84';'#fc8d59';'#ef6548';'#d7301f'; '#990000'];
% PLI is in shades of cold blue
pli_hex = ['#f1eef6'; '#d0d1e6'; '#a6bddb'; '#74a9cf'; '#3690c0'; '#0570b0'; '#034e7b'];

for step = ["01","10"]
    % data from the table
    table = readtable(('feature_weights_'+step+'.csv'));
    
    for mode = 1:height(table)

        epoch = string(table{mode, 1});
        feature = string(table{mode, 3});
        graph = string(table{mode, 2});
        mean_weight = table{mode, 4:81+4};
        std_weight = table{mode, 82+4:end};

        %replace missing values by 0 
        std_weight(isnan(std_weight))= 0;
        mean_weight(isnan(mean_weight))= 0;

        %% Plot the weights on the brain
        % Create the figure
        fixed_rois = get_fixed_rois();

        % To update the map we just need to change the Hex here by giving it 7
        % values, the interpl will then interpolate for the size required here 82
        if feature == "aec"
            map = get_color_map(aec_hex);
        end

        if feature == "pli"
            map = get_color_map(pli_hex);
        end

        %mean_weight
        make_figure_top(fixed_rois, map, mean_weight)
        savefig('figures/'+step+epoch+'_graph_'+graph+'_feature_'+feature+'_mean_top.fig')
        close()
        make_figure_sagittal(fixed_rois, map, mean_weight);
        savefig('figures/'+step+epoch+'_graph_'+graph+'_feature_'+feature+'_mean_sag.fig')
        close()

        %std_weight
        make_figure_top(fixed_rois, map, std_weight);
        savefig('figures/'+step+epoch+'_graph_'+graph+'_feature_'+feature+'_std_top.fig')
        close()
        
        make_figure_sagittal(fixed_rois, map, std_weight);
        savefig('figures/'+step+epoch+'_graph_'+graph+'_feature_'+feature+'_std_sag.fig')
        close()

    end    

end