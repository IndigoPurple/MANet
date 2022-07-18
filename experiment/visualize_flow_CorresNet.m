addpath flow_code_matlab/

close all

% base_path = 'LF_dataset_x8/LR(4,4)-REF(0,0)'; 
base_path = 'Flower_dataset_x8/LR(4,4)-REF(0,0)'; 

flow_vis_path = strcat( base_path, '/FlowNetS_multiscale_warp_decoder2_flow_vis/');
mkdir(flow_vis_path);

for i= 2   %1:1:100
    num = i;
   REF = imread(strcat( base_path, '/REF/', num2str(i), '.png'));
   GT = imread(strcat( base_path, '/GT/', num2str(i), '.png'));
   
%    imshow(REF);pause()
%    continue
%    
   filename = strcat( base_path, '/FlowNetS_multiscale_warp_decoder2/flow_', num2str(i), '.mat');
   disp(filename);
   
   load(filename)
  
   flow = flow0;
%    flow = flow_12;
% 
%    flow_c_1 = flow(:, 1:2:end, 1:2:end);
%    flow_c_2 = flow(:, 1:2:end, 2:2:end);
%    flow_c_3 = flow(:, 2:2:end, 1:2:end);
%    flow_c_4 = flow(:, 2:2:end, 2:2:end);
%    flow_c_mix = (flow_c_1+flow_c_2+flow_c_3+flow_c_4)/4;
   
   img = flowToColor(flow);
%    img_1 = flowToColor(flow_c_1);
%    img_2 = flowToColor(flow_c_2);
%    img_3 = flowToColor(flow_c_3);
%    img_4 = flowToColor(flow_c_4);
%    img_mix = flowToColor(flow_c_mix);   img_mix = imresize(img_mix,2);
   
   img_smooth = imgaussfilt(img,0.5);
  
   
    CorresNet_flow_index = permute(flow, [2,3,1]);
    save(fullfile(base_path,'FlowNetS_multiscale_warp_decoder2_flow_vis',strcat(num2str(i),'_flow_CorresNet.mat')),'CorresNet_flow_index');

    
%    figure(1); subplot(1,2,1); imshow(img); title('original'); 
%    subplot(1,2,2); imshow(GT); title('original'); 
%    pause();

%    figure(1); imagesc(HR2_conv1); title('original');
%    figure(2); imagesc(warp_21_conv1); title('warped');
   
%    figure(1); subplot(2,2,1);  imshow(flowToColor(flow1));
%    figure(1); subplot(2,2,2);  imshow(flowToColor(flow2));
%    figure(1); subplot(2,2,3);  imshow(flowToColor(flow3));
%    figure(1); subplot(2,2,4);  imshow(flowToColor(flow4)); 
   
%    figure(1);  imshow(img_mix);
%    figure(2); imshow(img_smooth);
%    figure(3); hist(double(reshape(flow(:,:,1),1,[])),30);

    imwrite(...
            flowToColor(flow0), fullfile(flow_vis_path, ...
                    strcat(num2str(i), '_scale0.png'))...
                    );
    imwrite(imresize(flowToColor(flow1*4),2.0, 'nearest'), fullfile(flow_vis_path, ...
                    strcat(num2str(i), '_scale1.png')));
    imwrite(imresize(flowToColor(flow2*3),4.0, 'nearest'), fullfile(flow_vis_path, ...
                    strcat(num2str(i), '_scale2.png')));
    imwrite(imresize(flowToColor(flow3*4),8.0, 'nearest'), fullfile(flow_vis_path, ...
                    strcat(num2str(i), '_scale3.png')))
    imwrite(imresize(flowToColor(flow4*32),16.0, 'nearest'), fullfile(flow_vis_path, ...
                    strcat(num2str(i), '_scale4.png')))
    imwrite(imresize(flowToColor(flow5*2),32.0, 'nearest'), fullfile(flow_vis_path, ...
                    strcat(num2str(i), '_scale5.png')))
    imwrite(imresize(flowToColor(flow6*2),64.0, 'nearest'), fullfile(flow_vis_path, ...
                    strcat(num2str(i), '_scale6.png')))

%     figure(1); imshow(imresize(flowToColor(flow4*5),8.0, 'nearest')); title('original');  
                
%     figure(100); imshow(GT);
%     figure(5);   imshow(permute(warp, [2,3,1]))

%    figure(4); subplot(2,2,1);  imshow(img_1);
%    figure(4); subplot(2,2,2);  imshow(img_2);
%    figure(4); subplot(2,2,3);  imshow(img_3);
%    figure(4); subplot(2,2,4);  imshow(img_4);
   
%    figure(3); subplot(2,2,1); hist(double(reshape(flow_c_1(:,:,1),1,[])),30);
%    figure(3); subplot(2,2,2); hist(double(reshape(flow_c_2(:,:,1),1,[])),30);
%    figure(3); subplot(2,2,3); hist(double(reshape(flow_c_3(:,:,1),1,[])),30);
%    figure(3); subplot(2,2,4); hist(double(reshape(flow_c_4(:,:,1),1,[])),30);

%    pause(0.01)
%    pause()
end