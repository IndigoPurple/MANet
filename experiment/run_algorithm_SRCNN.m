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


mkdir(fullfile(im_path,'SRCNN'))
%%
addpath compare_algorithms/SRCNN; 
model = 'compare_algorithms/SRCNN/model/9-5-5(ImageNet)/x4.mat'; %load(model);
run compare_algorithms/Patchmatch_ICCVW_VDSR/matconvnet/matlab/vl_setupnn

files = dir(fullfile(im_path,'GT','*.png'));
files = {files(:).name};
for name = files
   lr = im2double(Scene_imread_LR8(fullfile(im_path,'LR',name{:})));
%    lr = im2double(imread(fullfile(im_path,'LR',name{:})));
   lr_x4 = imresize(lr, 4, 'bicubic');
   if size(lr_x4,3)>1
        im_ycbcr = rgb2ycbcr(lr_x4);
        lr_x4_g = im_ycbcr(:, :, 1);
   end

   im_y_h = SRCNN(model, lr_x4_g);  
   
   clear im_h;
   if scale == 8
        im_y_h = imresize(im_y_h, 2, 'bicubic');
        im_h(:,:,1) = im_y_h;  
        im_h(:,:,2) = imresize(squeeze(im_ycbcr(:, :, 2)), 2, 'bicubic');
        im_h(:,:,3) = imresize(squeeze(im_ycbcr(:, :, 3)), 2, 'bicubic');
   else
        im_h(:,:,1) = im_y_h;  
        im_h(:,:,2) = squeeze(im_ycbcr(:, :, 2));
        im_h(:,:,3) = squeeze(im_ycbcr(:, :, 3));
   end
   im_h = ycbcr2rgb(im_h);
        
   imwrite(im_h,fullfile(im_path,'SRCNN',name{:}));
%    imshow(result);pause()
end

end