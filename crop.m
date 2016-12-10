M = csvread('pos_and_bm.csv');
Label = [];
count = 1;
for i = 1:1:135
infile = strcat('D:\上課相關\Fall 2016\767 Medical Image Analysis\processed_img\pp', num2str(i, '%03d'), '.png');
%infile = strcat('D:\上課相關\Fall 2016\767 Medical Image Analysis\project_idea\JPCLN', num2str(i, '%03d'), '.png');
im = imread(infile);

% while(1)
%     r = floor(rand*2048);
%     c = floor(rand*2048);
%     if ((r +127 < M(i,2)-127 || r -128 > M(i,2)+128) && ...
%         (c+127 < M(i,1)-127 || c-128 > M(i,1)+128) && ...
%         (r-127 >= 1) && (r+128 <= 2048) && (c-127 >= 1) && (c+128 <= 2048))
%         out = im(r-127:r+128, c-127:c+128);
%         break;
%     end
% end

% for k = 1:1:8
%     for j = 1:1:8
%         r = (k-1)*256+1;
%         c = (j-1)*256+1;
%         
%         %outfile = strcat('D:\上課相關\Fall 2016\767 Medical Image Analysis\767FinalProject\cropped_img\processed_n_patch', num2str(i,'%03d'),num2str(count, '%03d'), '.png');
%         %disp(size(out));
%         
%         
%         bound = M(i,3)*100/15/2/2;
%         if(M(i,1) + bound < r || M(i,1) - bound > r +255 || M(i,2) + bound > c+255 || M(i,2) - bound < c)
%             Label = [Label; 0];
%             %continue;
%         else
%             disp(i);
%             Label = [Label; 1];
%             out = im(r:r+255, c:c+255);
%             outfile = strcat('D:\上課相關\Fall 2016\767 Medical Image Analysis\767FinalProject\cropped_img\nodule_non_center', num2str(i,'%03d'),num2str(count, '%03d'), '.png');
%             imwrite(out,outfile);
%             count = count + 1;
%        end
%     end
% end
rshift = floor(rand*128);
cshift = floor(rand*128);
out = im(M(i,2)-rshift:M(i,2)+255-rshift, M(i,1)-cshift:M(i,1)+255-cshift);
outfile = strcat('D:\上課相關\Fall 2016\767 Medical Image Analysis\767FinalProject\cropped_img\nodule_non_center_processed', num2str(i,'%03d'), '.png');
imwrite(out,outfile);
%disp(size(crop_im));

end
disp(sum(Label(:)));
%filename = 'D:\上課相關\Fall 2016\767 Medical Image Analysis\767FinalProject\test_nodule_patch_label.csv';
%csvwrite(filename,Label);