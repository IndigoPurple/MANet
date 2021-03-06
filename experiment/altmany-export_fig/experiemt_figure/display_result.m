% imgpath = '../Flower_dataset_x8/LR(1,1)-REF(0,0)'; i = 34;   number = 1;
% imgpath = '../LF_dataset_x8/LR(3,3)-REF(0,0)'; i = 1;     number = 2;
imgpath = '../LF_dataset_x8/LR(1,1)-REF(0,0)'; i = 169;    number = 3;

% imgpath = '../Flower_dataset_x4/LR(1,1)-REF(0,0)'; i = 1;   number = 1;
% imgpath = '../LF_dataset_x4/LR(1,1)-REF(0,0)'; i = 91;     number = 4;
% imgpath = '../LF_dataset_x4/LR(1,1)-REF(0,0)'; i = 153;    number = 5;

mkdir(strcat(num2str(number),'/'));
for i=i
    EWDNet = imread(fullfile(imgpath, 'FlowNetS_multiscale_warp_decoder2', strcat(num2str(i), '.png')));
    
    figure(1);imshow(EWDNet)
    rect = getrect();   x = rect(1);  y = rect(2);  
    dx = max(rect(3), rect(4));  dy = max(rect(3), rect(4)); 
    rect = [x, y, dx, dy];
%     dx = rect(3);  dy = rect(4);
%     rect=[ rect(2), rect(1), rect(4), rect(3)];
%     rect=[ rect(1), rect(3), rect(2), rect(4)];

% imgpath = '../Flower_dataset_x8/LR(7,7)-REF(0,0)'; i = 2;
% x1 = 243; dx = 62; y1 = 64; dy = 64;  rect = [x1,dx,y1,dy];

GT = imread(fullfile(imgpath, 'GT', strcat(num2str(i), '.png')));
try
   LR = imread(fullfile(imgpath, 'LR_bicubic', strcat(num2str(i), '.png')));
catch exception
   LR = imread(fullfile(imgpath, 'LR_upsample', strcat(num2str(i), '.png')));
end
REF = imread(fullfile(imgpath, 'REF', strcat(num2str(i), '.png')));


SRCNN = imread(fullfile(imgpath, 'SRCNN', strcat(num2str(i), '.png')));
VDSR = imread(fullfile(imgpath, 'VDSR', strcat(num2str(i), '.png')));
MDSR = imread(fullfile(imgpath, 'MDSR', strcat(num2str(i), '.png')));
PM = imread(fullfile(imgpath, 'PatchMatch_lr_-0.008-8-bicubic', strcat(num2str(i), '.png')));
try
    SSNet = imread(fullfile(imgpath, 'BMVC_test', strcat(num2str(i), '.png')));
catch exception
    SSNet = 0;
end
EWDNet = imread(fullfile(imgpath, 'FlowNetS_multiscale_warp_decoder2', strcat(num2str(i), '.png')));

% pos_triangle = [183 297 302 250 316 297];
% pos_hexagon = [340 163 305 186 303 257 334 294 362 255 361 191];
% GT = insertShape(GT,'FilledPolygon',{pos_triangle,pos_hexagon},...
%     'Color', {'white','green'},'Opacity',0.7);
%     'Color', {'white','green'},'Opacity',0.7);

GT_c = GT(y:y+dy,x:x+dx,:);
LR = LR(y:y+dy,x:x+dx,:);
REF = REF(y:y+dy,x:x+dx,:);
SRCNN = SRCNN(y:y+dy,x:x+dx,:);
VDSR = VDSR(y:y+dy,x:x+dx,:);
MDSR = MDSR(y:y+dy,x:x+dx,:);
PM = PM(y:y+dy,x:x+dx,:);
try
    SSNet = SSNet(y:y+dy,x:x+dx,:);
end
EWDNet = EWDNet(y:y+dy,x:x+dx,:);

GT = insertShape(GT,'Rectangle', rect, 'LineWidth', 2);

figure(1);
imshow(GT);

figure(2);
subplot(3,3,1); imshow(SRCNN);
subplot(3,3,2); imshow(VDSR);
subplot(3,3,3); imshow(MDSR);
subplot(3,3,4); imshow(PM);
try
    subplot(3,3,5); imshow(SSNet);
end
subplot(3,3,6); imshow(EWDNet);
subplot(3,3,7); imshow(LR);
subplot(3,3,9); imshow(GT_c);

imwrite(LR, strcat(num2str(number),'/LR.png'));
imwrite(SRCNN, strcat(num2str(number),'/SRCNN.png'));
imwrite(VDSR, strcat(num2str(number),'/VDSR.png'));
imwrite(MDSR, strcat(num2str(number),'/MDSR.png'));

imwrite(PM, strcat(num2str(number),'/PM.png'));
try
    imwrite(SSNet, strcat(num2str(number),'/SSNet.png'));
end
imwrite(EWDNet, strcat(num2str(number),'/EWDNet.png'));

imwrite(GT_c, strcat(num2str(number),'/GT.png'));
imwrite(GT, strcat(num2str(number),'/GT_all.png'));

% pause()


end
