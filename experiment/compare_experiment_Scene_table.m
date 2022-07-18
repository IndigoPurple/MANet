addpath vif/;
addpath ifcvec_release/;
addpath matlabPyrTools/;

dataset_name = 'Scene_LF_pure_x8/';  %'Stanford_x8/';
file_list = 0:4;  %0:11
v_list = [1,3,5];  %[1]; %[1,3,5]; %[1,3,5];

% method_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch_lr_-0.008-8-bicubic', 'BMVC_test', 'FlowNetS_multiscale_warp_decoder2', 'FlowNetS_image_warp_decoder2'};
% legend_list = {'SRCNN', 'VDSR', 'MDSR', 'PatchMatch','BMVC', 'EWD-SRNet', 'WS-SRNet-p-20k'};

% method_list = {'Yuwang'};
% legend_list = {'Yuwang'};
% method_list = {'EWD-SRNet'};
% legend_list = {'EWD-SRNet'};
% method_list = {'EWD-SRNet-aug2'};
% legend_list = {'EWD-SRNet-aug2'};
method_list = {'MDSR'};
legend_list = {'MDSR'};
% method_list = {'PatchMatch_lr_-0.008-8-bicubic'};
% legend_list = {'PatchMatch'};
% method_list = {'BMVC_test'};
% legend_list = {'BMVC_test'};

% method_list = {'PatchMatch_lr_-0.008-8-bicubic','EWD-SRNet'};
% legend_list = {'PatchMatch','EWD-SRNet'};

% image_name = {'Amethyst.zip', 'Bracelet.zip', 'Chess.zip', 'Eucalyptus_Flowers.zip', 'Jelly_Beans.zip', ...
%             'Lego_Bulldozer.zip', 'Lego_Gantry_Self_Portrait.zip', 'Lego_Knights.zip', 'Lego_Truck.zip', ...
%             'Tarot_Cards_and_Crystal_Ball_large_angular_extent.zip', ...
%             'Tarot_Cards_and_Crystal_Ball_small_angular_extent.zip', 'The_Stanford_Bunny.zip', 'Treasure_Chest.zip'};

image_name = {'bikes', 'church', 'couch', 'mansion', 'statue'};


% psnr_table = zeros([view_points,length(method_list)]);
% ssim_table = zeros([view_points,length(method_list)]);
% vif_table = zeros([view_points,length(method_list)]);
p = zeros(length(file_list), 10);
for v = v_list
    im_path = strcat(dataset_name ,'LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
    for m = 1:length(method_list)
        psnr_list = [];
        ssim_list = [];
        vif_list = [];
        for f = file_list
           name = strcat(num2str(f),'.png');
           output_m = im2single(imread(fullfile(im_path, method_list{m}, name)));
           gt = im2single(imread(fullfile(im_path,'GT',name)));
           
           hh = 1712;  ww = 2616;
           output_m = output_m(1:hh, 1:ww,:);
           gt = gt(1:hh, 1:ww,:);
           
%            gt = gt(1:size(output_m,1), 1:size(output_m,2),:);
%            output_m = output_m(1:size(gt,1), 1:size(gt,2),:);
           
           psnr_ = compute_psnr(gt,output_m);
%            ssim_ = get_ssim(rgb2gray(uint8(255*output_m)), rgb2gray(uint8(255*gt)));
%            vif_ = ifcvec(rgb2gray(uint8(255*output_m)), rgb2gray(uint8(255*gt)));
           p(f+1,v) = psnr_;
%            fprintf('%s %s: %0.2f \n',legend_list{m}, image_name{f+1}, psnr_);
        end
    end
end

for f =  file_list  %[0,1,2,10,11]
    fprintf('%s: %0.2f / %0.2f / %0.2f \n', image_name{f+1}, p(f+1,1), p(f+1,3), p(f+1,5));
%     fprintf('%s: %0.2f / %0.2f / %0.2f \n', image_name{f+1}, 0.0, p(f+1,3), p(f+1,5));

end


fprintf('%s: %0.2f / %0.2f / %0.2f \n', 'average', mean(p(:,1)), mean(p(:,3)), mean(p(:,5)));
    
