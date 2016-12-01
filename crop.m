M = csvread('pos_and_bm.csv');
for i = 1:1:154
infile = strcat('D:\上課相關\Fall 2016\767 Medical Image Analysis\processed_img\pp', num2str(i, '%03d'), '.png');
im = imread(infile);
while(1)
    r = floor(rand*2048);
    c = floor(rand*2048);
    if ((r +127 < M(i,2)-127 || r -128 > M(i,2)+128) && ...
        (c+127 < M(i,1)-127 || c-128 > M(i,1)+128) && ...
        (r-127 >= 1) && (r+128 <= 2048) && (c-127 >= 1) && (c+128 <= 2048))
        out = im(r-127:r+128, c-127:c+128);
        break;
    end
end
%out = im(M(i,2)-127:M(i,2)+128, M(i,1)-127:M(i,1)+128);
%disp(size(crop_im));
outfile = strcat('D:\上課相關\Fall 2016\767 Medical Image Analysis\cropped_img\npatchp', num2str(i, '%03d'), '.png');
imwrite(out,outfile);
end