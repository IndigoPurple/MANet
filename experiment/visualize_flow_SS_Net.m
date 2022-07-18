% please refer to
% /fileserver/haitian/haitian_backup/HT_sr/SRResNet_After_BMVC/2_new_train_SS_Net_Flower_visualize.py
% for generating flow file


addpath flow_code_matlab/
scale = 8;

refpath = fullfile(strcat('Flower_dataset_x',num2str(scale),'/LR(',num2str(1),',',num2str(1),')-REF(0,0)'),'REF');
gtpath = fullfile(strcat('Flower_dataset_x',num2str(scale),'/LR(',num2str(1),',',num2str(1),')-REF(0,0)'),'GT');

files = dir(fullfile(refpath,'*.png'));
files = {files(:).name};

for name = files(11)  %files(1:end)
    name = {'11.png'};
    gt = im2single(imread(fullfile(gtpath,name{:})));
    num = strrep(name{:},'.png','');
    for v = 4
        im_path = strcat('Flower_dataset_x',num2str(scale),'/LR(',num2str(v),',',num2str(v),')-REF(0,0)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
        %% load flow
        load(fullfile(im_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_SSNet_avg_index.mat')));
        SS_Net_index = squeeze(SS_Net_average_index);
        SS_Net_index = permute(SS_Net_index,[2,3,1]);
        
        original_index = zeros(size(gt,1), size(gt,2),2);
        [index_X,index_Y] = meshgrid(1:size(gt,2),1:size(gt,1));
        original_index(:,:,1) = index_X;  original_index(:,:,2) = index_Y;
        SS_Net_flow_index = SS_Net_index - original_index;
        
        flow_color = flowToColor(SS_Net_flow_index);
        
        
%         figure(1);
%         subplot(1,2,1);imagesc(SS_Net_index(:,:,1));
%         subplot(1,2,2);imagesc(SS_Net_index(:,:,2));
%         figure(2);
%         subplot(1,2,1);imagesc(original_index(:,:,1));
%         subplot(1,2,2);imagesc(original_index(:,:,2));
%         figure(3)
%         subplot(1,2,1);imagesc(flow_index(:,:,1));
%         subplot(1,2,2);imagesc(flow_index(:,:,2));
        figure(4)
        imshow(flow_color);
        figure(5)
        imshow(gt);
        
        save(fullfile(im_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_flow_SSNet.mat')),'SS_Net_flow_index');
        imwrite(flow_color,fullfile(im_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_SSNet.png')));
    end
end
