function [map] = get_color_map(hex)
%% GET COLOR MAP given a hex code and a number of regions will give colormap
% input
% - hex: the hex array representing the colors
    vec = [ 0; 15; 30; 44; 68; 83 ; 100];
    raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
    N = 82;
    map = interp1(vec,raw,linspace(0,100,N),'pchip');
end