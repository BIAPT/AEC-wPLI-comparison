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