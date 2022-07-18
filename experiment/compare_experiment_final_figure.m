addpath vif/;
addpath ifcvec_release/;
addpath matlabPyrTools/;
addpath dorian3d-AcademicFigures/;

%% x8 SR
% method_list = {'FlowNetS_multiscale_warp_decoder2', 'BMVC_test', 'PatchMatch_lr_-0.008-8-bicubic', 'MDSR', 'VDSR', 'SRCNN'};
% legend_list = {'Ours', 'SS-Net', 'PatchMatch', 'MDSR', 'VDSR', 'SRCNN'};
% color_list = {[228,26,28]/255.0, [55,126,184]/255.0,  [77,175,74]/255.0, [152,78,163]/255.0, [255,127,0]/255.0, [166,86,40]/255.0};

% dataset_name = 'LF_dataset_x8/';
% filename = 'LFVideo_X8.pdf';
% ymin = 28; ymax = 46;

% dataset_name = 'Flower_dataset_x8/'; 
% filename = 'Flower_X8.pdf';
% ymin = 27; ymax = 42;
%% x4 SR
method_list = {'FlowNetS_multiscale_warp_decoder2', 'PatchMatch_lr_-0.008-8-bicubic', 'MDSR', 'VDSR', 'SRCNN'};
legend_list = {'Ours', 'PatchMatch', 'MDSR', 'VDSR', 'SRCNN'};
color_list = {[228,26,28]/255.0, [77,175,74]/255.0, [152,78,163]/255.0, [255,127,0]/255.0, [166,86,40]/255.0};

% dataset_name = 'Flower_dataset_x4/';  %'LF_dataset_x4/'; %'Flower_dataset_x8/';    %'LF_dataset_x8/';
% filename = 'Flower_X4.pdf';
% ymin = 32; ymax = 45;

dataset_name = 'LF_dataset_x4/';  %'LF_dataset_x4/'; %'Flower_dataset_x8/';    %'LF_dataset_x8/';
filename = 'LFVideo_X4.pdf';
ymin = 32; ymax = 45.5;


% dataset_name = 'LF_dataset_x8/';  %'LF_dataset_x4/'; %'Flower_dataset_x8/';    %'LF_dataset_x8/';
% filename = 'LFVideo_X8.pdf';
% method_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch_lr_-0.008-8-bicubic', 'BMVC_test', 'FlowNetS_multiscale_warp_decoder2'};
% legend_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch','SS-Net', 'Ours'};

% method_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch_lr_-0.008-8-bicubic', 'BMVC_test', 'FlowNetS_multiscale_warp_decoder2', 'FlowNetS_image_warp_decoder2'};
% legend_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch','BMVC', 'EWD-SRNet', 'WS-SRNet-p-20k'};
% method_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch_lr_-0.008-8-bicubic', 'FlowNetS_multiscale_warp_decoder2'};
% legend_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch', 'Ours'};
% method_list = {'FlowNetS_multiscale_warp_decoder2', 'WS-SRNet-p-20k', 'WS-SRNet-p-40k'};
% legend_list = {'EWD-SRNet', 'WS-SRNet-p-20k', 'WS-SRNet-p-40k'};

view_points = 7;
psnr_table = zeros([view_points,length(method_list)]);
psnr_table_std = zeros([view_points,length(method_list)]);

for v = 1:7
    disp(v)
    im_path = strcat(dataset_name ,'LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
% for v = 4:5  %1:5
%     im_path = strcat('Sintel_dataset/LR(x-',num2str(v),')-REF(x)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';

%     files = dir(fullfile(strcat(dataset_name ,'LR(',num2str(7),',',num2str(7),')-REF(0,0)'),'GT','*.png'));
%     files = dir(fullfile(im_path,'BMVC_test','*.png'));
    files = dir(fullfile(im_path,'GT','*.png'));
    files = {files(:).name};

    for m = 1:length(method_list)
        psnr_list = [];
        for name = files(1:end) %(1:1)    %files(1:end-50) 
           gt = im2single(imread(fullfile(im_path,'GT',name{:})));
           output_m = im2single(imread(fullfile(im_path, method_list{m}, name{:})));
           output_m = output_m(1:size(gt,1), 1:size(gt,2),:);
           
           psnr_list = [psnr_list, compute_psnr(gt,output_m)];
        end
        psnr_table(v,m) = mean(psnr_list);
        psnr_table_std(v,m) = std(psnr_list);
    end
end

%% show figure
close all;
afigure(aconfig('LineStyles', '-') );

hold on;
for m = 1:length(method_list)
    plot(psnr_table(:,m),'Color',color_list{m});
end
legend(legend_list);
xlabel('Disparity');
ylabel('PSNR (dB)');

xlim([1,7]);
ylim([ymin, ymax]);
set(gca,'xtick',1:7); 
set(gca,'xticklabel',{'(1,1)'; '(2,2)'; '(3,3)'; '(4,4)'; '(5,5)'; '(6,6)'; '(7,7)'});
set(gca,'ytick',ymin:2:ymax); 
set(gca, 'FontSize', 17)
grid minor
% grid on
% set(gcf,'PaperPositionMode','auto')

fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf,filename,'-dpdf')
