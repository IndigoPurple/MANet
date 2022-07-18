function [ im ] = Scene_imread( filename )
H = 1752; W = 2672;
% H = 1712;  W = 2616;
% H = 1712;  W = 2616;
im = imread(filename);
im = im(1:H,1:W,:);
end

