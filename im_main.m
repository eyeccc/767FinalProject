for i = 1:1:93
    str = 'D:\¤W½Ò¬ÛÃö\Fall 2016\767 Medical Image Analysis\project_idea\JPCNN';
    infile = strcat(str, num2str( i, '%03d' ),'.png');
    im = imread(infile);
    out = preprocess(im);
    outfile = strcat('processed_img\pn', num2str(i, '%03d'), '.png');
    imwrite(out,outfile);
end