%% Yacine Mahdid July 3 2020
% This script is used to plot the weights from the trained ML classifier
% onto the AAL brain atlas.
% 

%% Experimental Variable
% AEC is in shades of hot red-orange
aec_hex = ['#fef0d9';'#fdd49e';'#fdbb84';'#fc8d59';'#ef6548';'#d7301f'; '#990000'];
% PLI is in shades of cold blue
pli_hex = ['#f1eef6'; '#d0d1e6'; '#a6bddb'; '#74a9cf'; '#3690c0'; '#0570b0'; '#034e7b'];

step = ["01"]
% data from the table
table = readtable(('features_step'+step+'.csv'));

graphs =["aec","wpli"];
epochs = ["eyes_closed_1","induction","emergence_first",...
    "emergence_last","eyesclosed_8"];

for graph = [1, 2]
    for epoch = [1,2,3,4,5]

        tmp = table(table.epoch == epoch,:);
        tmp2 = tmp(tmp.graph == graph,:);

        values = tmp2(:, 6:87);
        toplot = mean(values{:,:},1)

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
        plot_brain_top(fixed_rois, map, toplot);
 
        fixed_rois = get_fixed_rois();

        make_figure_top(fixed_rois, jet(100), toplot)
        colorbar
        caxis([-10 10])
        
        
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


function plot_brain(fixed_rois, map, cur_weights, all_weights, region_range, view_coordinate)
    
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



