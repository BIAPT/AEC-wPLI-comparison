function aal_brain(roi_range,weights,roi, map)
    % fc = functional connectivity matrix or vector
    % roi = volume matrix of ROI regions, 3dims x no. ROIs  
    
    % Create the figure
    figure
    axis ([0,100,0,50,0,100])
    
    map = [[0,0,0]; [1 0 0]]
    c = colormap(map);
    hold all

    colorindex = weights(roi_range);

    for reg = 1:length(roi_range)
        roisurf=isosurface(roi(:,:,:,reg),0.5);
        h = trisurf(roisurf.faces,roisurf.vertices(:,1),roisurf.vertices(:,2),roisurf.vertices(:,3));
        set(h,'facecolor',c(colorindex(reg),:),'facealpha',0.8,'LineWidth',0.1,'LineStyle','none');
    end

    set(gca,'view',[-90 90])
    axis equal
    axis off
    set(gcf,'color','white')
end