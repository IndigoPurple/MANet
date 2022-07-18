
test_img_file = 'test/Knights/out_00_00_-381.909271_1103.376221.png';
ref_img_file = 'test/Knights/out_00_10_-383.248383_1017.387878.png';

% test_img_file = 'test/Amethyst/out_00_00_-399.772888_671.237122_.png';
% ref_img_file = 'test/Amethyst/out_00_10_-399.770660_770.764648_.png';

up_scale = 3;

    %% patch match
    addpath('PatchMatch_new');
    % parameters for patch match super-resolution
    params.lpatch_size = 5;
    params.SR_factor = 8;
    params.nn = 4;
    params.search_range = 8;
    params.beta = -0.008;
    
    ref = imread(ref_img_file);
%     ref = single(ref)/255;
    ref = repmat(rgb2gray(ref),[1,1,3]);
    
    gt_h = imread(test_img_file);
%     gt_h = single(gt_h)/255;
    gt_h = repmat(rgb2gray(gt_h),[1,1,3]);
    
    gt_l = imresize(gt_h,1/params.SR_factor,'bicubic');
    
    ref_cell = cell(1,1); ref_cell{1,1} = ref;
    % get dictionary
    [lf,lf_inds] = GetLFeat2(ref_cell,params); %ref
    % patch matching in low resolution
    [im_h_PM1, im_h_PM2] = InferSR2(lf,lf_inds,ref_cell,gt_l,params); %ref
    figure(1);
    subplot(1,2,1);
    imshow(im_h_PM1);
    subplot(1,2,2);
    imshow(im_h_PM2);
%     im_gnd_tmp = im_gnd;
%     draw_figure_difference(uint8(im_gnd_tmp*255),uint8(im_h_PM*255), 3,4,'gt','PatchMatch' );