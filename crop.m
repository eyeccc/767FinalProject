M = csvread('pos_and_bm.csv');
for i = 1:1:154
infile = strcat('D:\上課相關\Fall 2016\767 Medical Image Analysis\processed_img\pp', num2str(i, '%03d'), '.png');
im = imread(infile);
out = im(M(i,2)-127:M(i,2)+128, M(i,1)-127:M(i,1)+128);
%disp(size(crop_im));
outfile = strcat('D:\上課相關\Fall 2016\767 Medical Image Analysis\cropped_img\c', num2str(i, '%03d'), '.png');
imwrite(out,outfile);
end