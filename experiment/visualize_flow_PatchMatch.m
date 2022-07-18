addpath flow_code_matlab/
scale = 8;
addpath compare_algorithms/Patchmatch_ICCVW_VDSR/PatchMatch
% addpath compare_algorithms/Patchmatch_ICCVW_VDSR/PatchMatch_new

params.lpatch_size = 8;
params.SR_factor = scale;
params.nn = 1;
params.search_range = 16;
params.beta = -0.008;

refpath = fullfile(strcat('Flower_dataset_x',num2str(scale),'/LR(',num2str(1),',',num2str(1),')-REF(0,0)'),'REF');
gtpath = fullfile(strcat('Flower_dataset_x',num2str(scale),'/LR(',num2str(1),',',num2str(1),')-REF(0,0)'),'GT');

files = dir(fullfile(refpath,'*.png'));
files = {files(:).name};


for name = files(11)  %files(1:end)
    num = strrep(name{:},'.png','');
    name = {'11.png'};
    num = '11'
    disp(num)
    ref = im2single(imread(fullfile(refpath,name{:})));
    gt = im2single(imread(fullfile(gtpath,name{:})));
%     figure(1);imshow(gt);pause();continue;
    % get dictionary
    [lf,lf_inds] = GetLFeat2(ref,params); %ref
    for v = 4
        im_path = strcat('Flower_dataset_x',num2str(scale),'/LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
        mkdir(fullfile(im_path,'PatchMatch_lr_-0.008-8-bicubic'))
        lr = im2single(imread(fullfile(im_path,'LR',name{:})));
%         gt_lr_bilinear = im2single(imread(fullfile(im_path,'GT',name{:})));
%         gt_lr_bilinear = imresize(gt_lr_bilinear, 1/scale, 'bilinear');
        
        % patch matching original
%         im_sr = InferSR2(lf,lf_inds,ref,lr,params); 
        [im_sr, im_sr_index] = InferSR2_visualize_flow(lf,lf_inds,ref,lr,params); 
        original_index = zeros(size(im_sr,1), size(im_sr,2),2);
        [index_X,index_Y] = meshgrid(1:size(im_sr,2),1:size(im_sr,1));
        original_index(:,:,1) = index_X;
        original_index(:,:,2) = index_Y;
        flow_index = im_sr_index - original_index;
        flow_color = flowToColor(flow_index);
        
        figure(1);
        subplot(1,2,1);imagesc(im_sr_index(:,:,1));
        subplot(1,2,2);imagesc(im_sr_index(:,:,2));
        figure(2);
        subplot(1,2,1);imagesc(original_index(:,:,1));
        subplot(1,2,2);imagesc(original_index(:,:,2));
        figure(3)
        subplot(1,2,1);imagesc(flow_index(:,:,1));
        subplot(1,2,2);imagesc(flow_index(:,:,2));
        figure(4)
        imshow(flow_color);
        figure(5)
        imshow(gt);
        
        PatchMatch_flow_index = flow_index;
        save(fullfile(im_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_flow_PatchMatch.mat')),'PatchMatch_flow_index');
        imwrite(flow_color,fullfile(im_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_PatchMatch_NN1.png')));

    end
end
