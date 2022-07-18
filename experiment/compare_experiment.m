% method_list = {'PatchMatch_lr', 'FlowNetS_warping', 'FlowNetS_warping_C5', 'FlowNetS_warp_decoder2', 'FlowNetS_multiscale_warp_decoder2'};
% legend_list = {'PatchMatch', 'FlowNetS warp', 'FlowNetS warp + C5', 'FlowNetS warp + decoder2', 'FlowNetS multiscale warp + decoder2'};

method_list = {'FlowNetS_warping', 'FlowNetS_warping_C5', 'FlowNetS_warp_decoder2', 'FlowNetS_multiscale_warp_decoder2','BMVC'};
legend_list = {'FlowNetS warp', 'FlowNetS warp + C5', 'FlowNetS warp + decoder2', 'FlowNetS multiscale warp + decoder2','BMVC'};

psnr_table = zeros([7,length(method_list)]);
for v = 1:7  %1:7
    im_path = strcat('LF_dataset_x8/LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
% for v = 4:5  %1:5
%     im_path = strcat('Sintel_dataset/LR(x-',num2str(v),')-REF(x)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';

    files = dir(fullfile(im_path,'GT','*.png'));
    files = {files(:).name};

    for m = 1:length(method_list)
        psnr_list = [];
        for name = files(1:end) %files(1:end) 
    %      disp(name);
           gt = im2single(imread(fullfile(im_path,'GT',name{:})));
    %        lr = im2single(imread(fullfile(im_path,'LR',name{:})));
    %        lr_bicubic = im2single(imread(fullfile(im_path,'LR_upsample',name{:})));
    %        ref = im2single(imread(fullfile(im_path,'REF',name{:})));
           output_m = im2single(imread(fullfile(im_path, method_list{m}, name{:})));
           psnr_list = [psnr_list, compute_psnr(gt,output_m)];
        end
        psnr_table(v,m) = mean(psnr_list);
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
xlim([1,7]);
ylim([34.,41.5]);