imgpath = '../Flower_dataset_x8/LR(7,7)-REF(0,0)';
i = 2;x = 244; y = 62; s = 128;
% i = 55;x = 55; y = 92; s = 128;

LR = imread(fullfile(imgpath, 'LR_bicubic', strcat(num2str(i), '.png')));
REF = imread(fullfile(imgpath, 'REF', strcat(num2str(i), '.png')));
MDSR = imread(fullfile(imgpath, 'MDSR', strcat(num2str(i), '.png')));
result = imread(fullfile(imgpath, 'FlowNetS_multiscale_warp_decoder2', strcat(num2str(i), '.png')));

% imshow(LR); pause();
% imshow(REF); pause();
% imshow(MDSR); pause();
% imshow(result); pause();

LR = LR(y:y+s,x:x+s,:);
REF = REF(y:y+s,x:x+s,:);
MDSR = MDSR(y:y+s,x:x+s,:);
result = result(y:y+s,x:x+s,:);

imwrite(LR, 'cropped/LR.png');
imwrite(REF, 'cropped/REF.png');
imwrite(MDSR, 'cropped/MDSR.png');
imwrite(result, 'cropped/result.png');

imshow(LR); pause();
imshow(REF); pause();
imshow(MDSR); pause();
imshow(result); pause();
