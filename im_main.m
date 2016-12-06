count = 0;
out = [];
outtmp = [];
for i = 1:1:64
    %str = '/Users/waster/Downloads/bone_shadow_eliminated_JSRT_2013-04-19/JPCLN';
    str = '/Users/waster/767FinalProject/outfile'
    infile = strcat(str, num2str( i-1),'.png');%, '%03d' 
    im = imread(infile);
    outtmp = [outtmp, im];
    count = count + 1;
    if(count >= 8)
    	out = [out; outtmp];
    	outtmp = [];
    	count = 0;
    end
    

    %out = preprocess(im);
    %outfile = strcat('/Users/waster/Desktop/767img/pp', num2str(i, '%03d'), '.png');
    %imwrite(out,outfile);
    %[featureVector,hogVisualization] = extractHOGFeatures(im);
    %figure;
    %imshow(img);
    %hold on;
    %plot(hogVisualization);
end

imshow(out);