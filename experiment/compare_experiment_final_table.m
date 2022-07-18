addpath vif/;
addpath ifcvec_release/;
addpath matlabPyrTools/;

dataset_name = 'Flower_dataset_x4/';     %'LF_dataset_x8/';  %'LF_dataset_x4/'; %'Flower_dataset_x8/';    %'Flower_dataset_x4/';
% method_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch_lr_-0.008-8-bicubic', 'BMVC_test', 'FlowNetS_multiscale_warp_decoder2', 'FlowNetS_image_warp_decoder2'};
% legend_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch','BMVC', 'EWD-SRNet', 'WS-SRNet-p-20k'};
% method_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch_lr_-0.008-8-bicubic', 'BMVC_test', 'FlowNetS_multiscale_warp_decoder2'};
% legend_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch','BMVC', 'EWD-SRNet'};
% method_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch_lr_-0.008-8-bicubic', 'FlowNetS_multiscale_warp_decoder2'};
% legend_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch', 'EWD-SRNet'};
% method_list = {'FlowNetS_multiscale_warp_decoder2', 'WS-SRNet-p-20k', 'WS-SRNet-p-40k'};
% legend_list = {'EWD-SRNet', 'WS-SRNet-p-20k', 'WS-SRNet-p-40k'};

% method_list = {'FlowNetS_multiscale_warp_decoder2'};
% legend_list = {'EWD-SRNet'};
% method_list = {'FlowNetS_multiscale_warp_decoder2-enlarge1'};
% legend_list = {'EWD-SRNet-enlarge'};
% method_list = {'WS-SRNet-j-20k'};
% legend_list = {'WS-SRNet-j-20k'};
% method_list = {'WS-SRNet-p-20k'};
% legend_list = {'WS-SRNet-p-20k'};
% method_list = {'FlowNetS_multiscale_warp_decoder2', 'WS-SRNet-j-20k', 'WS-SRNet-p-20k'};
% legend_list = {'EWD-SRNet', 'WS-SRNet-j-20k', 'WS-SRNet-p-20k'};
method_list = {'SRGAN'};
legend_list = {'SRGAN'};

view_points = 7;
psnr_table = zeros([view_points,length(method_list)]);
psnr_table_std = zeros([view_points,length(method_list)]);
ssim_table = zeros([view_points,length(method_list)]);
ssim_table_std = zeros([view_points,length(method_list)]);
vif_table = zeros([view_points,length(method_list)]);
vif_table_std = zeros([view_points,length(method_list)]);

for v = [1,7]
    im_path = strcat(dataset_name ,'LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
% for v = 4:5  %1:5
%     im_path = strcat('Sintel_dataset/LR(x-',num2str(v),')-REF(x)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';

    files = dir(fullfile(strcat(dataset_name ,'LR(',num2str(7),',',num2str(7),')-REF(0,0)'),'GT','*.png'));
%     files = dir(fullfile(im_path,'BMVC_test','*.png'));
%     files = dir(fullfile(im_path,'GT','*.png'));
    files = {files(:).name};

    for m = 1:length(method_list)
        disp(legend_list{m})
        psnr_list = [];
        ssim_list = [];
        vif_list = [];
        for name = files(1:end) %(1:1)    %files(1:end-50) 
           gt = im2single(imread(fullfile(im_path,'GT',name{:})));
           output_m = im2single(imread(fullfile(im_path, method_list{m}, name{:})));
           output_m = output_m(1:size(gt,1), 1:size(gt,2),:);
           
%            figure(m);imshow(output_m);title(legend_list{m}); pause(0.01);
           
           psnr_list = [psnr_list, compute_psnr(gt,output_m)];
           ssim_list = [ssim_list, get_ssim(rgb2gray(uint8(255*output_m)), rgb2gray(uint8(255*gt)))];
           vif_list = [vif_list, ifcvec(rgb2gray(uint8(255*output_m)), rgb2gray(uint8(255*gt)))];
        end
        psnr_table(v,m) = mean(psnr_list);
        psnr_table_std(v,m) = std(psnr_list);
        ssim_table(v,m) = mean(ssim_list);
        ssim_table_std(v,m) = std(ssim_list);
        vif_table(v,m) = mean(vif_list);
        vif_table_std(v,m) = std(vif_list);
    end
end

%% display number
v_list = [1,7];
for v = v_list
    fprintf('%s, viewpoint %d\n', dataset_name, v);
    for m = 1:length(legend_list)
        fprintf('%s : %.2f / %.2f / %.2f \n', legend_list{m}, psnr_table(v,m), ssim_table(v,m), vif_table(v,m));
    end
end