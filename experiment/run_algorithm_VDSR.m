scale = 8;

% for v = 1:7
% im_path = strcat('Flower_dataset_x',num2str(scale),'/LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
% for v = 1:7
% im_path = strcat('LF_dataset_x',num2str(scale),'/LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
% for v = 1:5
% im_path = strcat('Sintel_dataset/LR(x-',num2str(v),')-REF(x)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
% for v = [1,3,5]
% im_path = strcat('Stanford_x',num2str(scale),'/LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
for v = [1,3,5]
im_path = strcat('Scene_LF_pure_x',num2str(scale),'/LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
mkdir(fullfile(im_path,'VDSR'))
%%
addpath compare_algorithms/Patchmatch_ICCVW_VDSR; 
model = 'compare_algorithms/Patchmatch_ICCVW_VDSR/model/VDSR/VDSR_Official.mat'; load(model);
run compare_algorithms/Patchmatch_ICCVW_VDSR/matconvnet/matlab/vl_setupnn

files = dir(fullfile(im_path,'GT','*.png'));
files = {files(:).name};
for name = files
   lr = im2double(Scene_imread_LR8(fullfile(im_path,'LR',name{:})));
%    lr = im2double(imread(fullfile(im_path,'LR',name{:})));
%    try 
%     lr_upsample = im2double(imread(fullfile(im_path,'LR_bicubic',name{:})));
%    catch
%     lr_upsample = im2double(imread(fullfile(im_path,'LR_upsample',name{:})));
%    end
%    ref = im2double(imread(fullfile(im_path,'REF',name{:})));
   %
   
   result = VDSR(model, lr, scale);
   
   imwrite(result,fullfile(im_path,'VDSR',name{:}));
%    imshow(result);pause()
end

end