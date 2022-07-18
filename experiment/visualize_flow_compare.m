base_path = 'Flower_dataset_x8/LR(4,4)-REF(0,0)'; 
base_path_fake = 'Flower_dataset_x8/LR(4,4)-REF(0,0)'; 
for i = 11  %1:100
   num = num2str(i);
   disp(num); 
    
   GT = im2single(imread(strcat( base_path_fake, '/GT/', num2str(i), '.png')));
   GT = AdjustTone(GT, 1.7) ;
   
%    imshow(GT); pause();continue;
   
   CorresNet = im2single(imread(strcat( base_path_fake, '/FlowNetS_multiscale_warp_decoder2/', num2str(i), '.png')));
   CorresNet = AdjustTone(CorresNet, 1.7) ;
   
   MDSR = im2single(imread(strcat( base_path_fake, '/MDSR/', num2str(i), '.png')));
   MDSR = AdjustTone(MDSR, 1.7) ;
   
%    SRGAN = im2single(imread(strcat( base_path_fake, '/SRGAN/', num2str(i), '.png')));
%    SRGAN = AdjustTone(SRGAN, 1.7) ;
   
   imwrite(GT,fullfile(base_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'.png')));
   imwrite(MDSR,fullfile(base_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_MDSR.png')));
   imwrite(CorresNet,fullfile(base_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_CorresNet.png')));
%    imwrite(SRGAN,fullfile(base_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_SRGAN.png')));
%    return 
   load(fullfile(base_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_flow_PatchMatch.mat')));
   load(fullfile(base_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_flow_SSNet.mat')));
   load(fullfile(base_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_flow_CorresNet.mat')));
   
   flow_all = cat(1,PatchMatch_flow_index, SS_Net_flow_index, CorresNet_flow_index);
   flow_all_color = flowToColor(flow_all);

   flow_PM = flowToColor(PatchMatch_flow_index);
   flow_SS_Net = flowToColor(SS_Net_flow_index);
   flow_CorresNet = flowToColor(CorresNet_flow_index);
   
%    flow_all(:,:,1) = 
   
   figure(1); imshow(flow_all_color);
   imwrite(flow_PM, fullfile(base_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_flow_PatchMatch.png')));
   imwrite(flow_SS_Net, fullfile(base_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_flow_SSNet.png')));
   imwrite(flow_CorresNet, fullfile(base_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num,'_flow_CorresNet.png')));
end