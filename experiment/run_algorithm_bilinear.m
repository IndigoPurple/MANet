scale = 8;
for v = [1,3,5]
    im_path = strcat('Scene_LF_pure_x', num2str(scale) ,'/LR(',num2str(v),',',num2str(v),')-REF(0,0)');
%     im_path = strcat('Stanford_x', num2str(scale) ,'/LR(',num2str(v),',',num2str(v),')-REF(0,0)');
    
% for v = 1:7
%     im_path = strcat('Flower_dataset_x',num2str(scale),'/LR(',num2str(v),',',num2str(v),')-REF(0,0)');
% for v = 1:5
    % im_path = strcat('Sintel_dataset/LR(x-',num2str(v),')-REF(x)');    %'LF_dataset_x8/LR(1,1)-REF(0,0)';
    mkdir(fullfile(im_path,'Bilinear'))
    %%
    disp(v)
    files = dir(fullfile(im_path,'GT','*.png'));
    files = {files(:).name};
    for name = files
       gt = im2double(imread(fullfile(im_path,'GT',name{:})));
       lr_bilinear = imresize( imresize(gt, 1/scale, 'bilinear', 'Antialiasing', false), scale, 'bilinear', 'Antialiasing', false);
       imwrite(lr_bilinear,fullfile(im_path,'Bilinear',name{:}));
    end
end