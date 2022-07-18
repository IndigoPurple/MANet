for v = 1:7
    im_path = strcat('Flower_dataset_x',num2str(scale),'/LR(',num2str(v),',',num2str(v),')-REF(0,0)');
% for v = 4:5 %1:5
%     im_path = strcat('Sintel_dataset/LR(x-',num2str(v),')-REF(x)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';

mkdir(fullfile(im_path,'PatchMatch_lr'))
mkdir(fullfile(im_path,'PatchMatch_VDSR'))

scale = 8;

%%
addpath compare_algorithms/Patchmatch_ICCVW_VDSR;
addpath compare_algorithms/Patchmatch_ICCVW_VDSR/PatchMatch
model = 'compare_algorithms/Patchmatch_ICCVW_VDSR/model/VDSR/VDSR_Official.mat'; load(model);
run compare_algorithms/Patchmatch_ICCVW_VDSR/matconvnet/matlab/vl_setupnn

params.lpatch_size = 8;
params.SR_factor = scale;
params.nn = 4;
params.search_range = 8;
params.beta = -0.008;

files = dir(fullfile(im_path,'GT','*.png'));
files = {files(:).name};
for name = files(1:50)  %files(1:end) 
   disp(name);
   gt = im2single(imread(fullfile(im_path,'GT',name{:})));
   lr = im2single(imread(fullfile(im_path,'LR',name{:})));
   lr_bicubic = im2single(imread(fullfile(im_path,'LR_upsample',name{:})));
   ref = im2single(imread(fullfile(im_path,'REF',name{:})));

   %
   lr_VDSR = VDSR(model, lr, scale);
   ref_l = imresize(ref, 1/scale, 'bicubic'); ref_l_VDSR = VDSR(model, ref_l, scale);
   %
   [result, result_ICCVW17] = PM(lr, ref, lr_VDSR, ref_l_VDSR, params);

   %%
%    gt_h=gt;
% %     gt_h = repmat(rgb2gray(gt_h),[1,1,3]); 
%     gt_l = imresize(gt_h,1/params.SR_factor,'bicubic');
% %     ref_cell = cell(1,1); ref_cell{1,1} = ref;
%     % get dictionary
%     [lf,lf_inds] = GetLFeat2(ref,params); %ref
%     % patch matching in low resolution
%     result = InferSR2(lf,lf_inds,ref,gt_l,params); %ref
%     figure(1);  imshow(result); pause()
    
%    imwrite(result,fullfile(im_path,'PatchMatch_lr',name{:}));
   imwrite(result,fullfile(im_path,'PatchMatch_lr_-0.008',name{:}));
   
   
%    imwrite(result_ICCVW17,fullfile(im_path,'PatchMatch_VDSR',name{:}));
   compute_psnr(gt,result)
%    compute_psnr(gt,result_ICCVW17)
end

end