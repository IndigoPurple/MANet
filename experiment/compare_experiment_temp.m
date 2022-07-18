% method_list = {'PatchMatch_lr','PatchMatch_lr_-0.008','PatchMatch_lr_-0.004','PatchMatch_lr_BMVC'};
% legend_list = {'PatchMatch_lr','PatchMatch_lr_-0.008','PatchMatch_lr_-0.004','PatchMatch_lr_BMVC'};

% method_list = {'PatchMatch_lr_-0.004','PatchMatch_lr_BMVC'};
% legend_list = {'PatchMatch_lr_-0.004','PatchMatch_lr_BMVC'};

% method_list = {'BMVC'};
% legend_list = {'BMVC'};

% method_list = {'PatchMatch_lr_-0.008-8-bicubic'};
% legend_list = {'PatchMatch_lr_-0.008-8-bicubic'};

% method_list = {'BMVC','BMVC_test'};
% legend_list = {'BMVC','BMVC_test'};

% method_list = {'PatchMatch_lr_-0.008-8-bicubic', 'FlowNetS_warping', 'FlowNetS_warping_C5', 'FlowNetS_warp_decoder2', 'FlowNetS_multiscale_warp_decoder2','BMVC_test'};
% legend_list = {'PatchMatch','FlowNetS warp', 'FlowNetS warp + C5', 'FlowNetS warp + decoder2', 'FlowNetS multiscale warp + decoder2','BMVC'};

method_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch_lr_-0.008-8-bicubic', 'BMVC_test', 'FlowNetS_multiscale_warp_decoder2'};
legend_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch','BMVC', 'EWD-SRNet'};

psnr_table = zeros([7,length(method_list)]);
psnr_table_std = zeros([7,length(method_list)]);
for v = 1:7
    im_path = strcat('LF_dataset_x8/LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
% for v = 4:5  %1:5
%     im_path = strcat('Sintel_dataset/LR(x-',num2str(v),')-REF(x)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';

    files = dir(fullfile(strcat('LF_dataset_x8/LR(',num2str(7),',',num2str(7),')-REF(0,0)'),'BMVC_test','*.png'));
%     files = dir(fullfile(im_path,'BMVC_test','*.png'));
%     files = dir(fullfile(im_path,'GT','*.png'));
    files = {files(:).name};

    for m = 1:length(method_list)
        psnr_list = [];
        for name = files(1:end) %(1:1)    %files(1:end-50) 
           gt = im2single(imread(fullfile(im_path,'GT',name{:})));
           output_m = im2single(imread(fullfile(im_path, method_list{m}, name{:})));
           output_m = output_m(1:size(gt,1), 1:size(gt,2),:);
           
%            figure(m);imshow(output_m);title(legend_list{m}); pause(0.01);
           
           psnr_list = [psnr_list, compute_psnr(gt,output_m)];
        end
        psnr_table(v,m) = mean(psnr_list);
        psnr_table_std(v,m) = std(psnr_list);
%         mean(psnr_list)
    end
end

figure(15); hold on;
for m = 1:length(method_list)
    plot(psnr_table(:,m));
%     legend(legend_list{m});
%     title(method_list{m});
end
legend(legend_list);
% xlim([1,7]);
% ylim([34.,41.5]);