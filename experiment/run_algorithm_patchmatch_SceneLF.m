scale = 8;
addpath compare_algorithms/Patchmatch_ICCVW_VDSR/PatchMatch
% addpath compare_algorithms/Patchmatch_ICCVW_VDSR/PatchMatch_new

params.lpatch_size = 8;
params.SR_factor = scale;
params.nn = 4;
params.search_range = 16;
params.beta = -0.008;

refpath = fullfile(strcat('Scene_LF_pure_x',num2str(scale),'/LR(',num2str(1),',',num2str(1),')-REF(0,0)'),'REF');
files = dir(fullfile(refpath,'*.png'));
files = {files(:).name};

for name = files(3:end)
    ref = im2single(imread(fullfile(refpath,name{:})));
    % get dictionary
    [lf,lf_inds] = GetLFeat2(ref,params); %ref
    for v = [1,3,5]
        if v == 5
            params.search_range = 32;
        else 
            params.search_range = 16;
        end
            
        im_path = strcat('Scene_LF_pure_x',num2str(scale),'/LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
        mkdir(fullfile(im_path,'PatchMatch_lr_-0.008-8-bicubic'))
        lr = im2single(imread(fullfile(im_path,'LR',name{:})));
%         gt_lr_bilinear = im2single(imread(fullfile(im_path,'GT',name{:})));
%         gt_lr_bilinear = imresize(gt_lr_bilinear, 1/scale, 'bilinear');
        
        % patch matching original
        im_sr = InferSR2(lf,lf_inds,ref,lr,params);
        imwrite(im_sr,fullfile(im_path,'PatchMatch_lr_-0.008-8-bicubic',name{:}));
    end
end
