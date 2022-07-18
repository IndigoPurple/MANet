function [ im ] = Scene_imread_LR8( filename )
H = 1712; W = 2616;
H = H/8; W = W/8;
im = imread(filename);
im = im(1:H,1:W,:);
end

