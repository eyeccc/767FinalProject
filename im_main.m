for i = 1:1:154
    str = '/Users/waster/Downloads/bone_shadow_eliminated_JSRT_2013-04-19/JPCLN';
    infile = strcat(str, num2str( i, '%03d' ),'.png');
    im = imread(infile);
    out = preprocess(im);
    outfile = strcat('/Users/waster/Desktop/767img/pp', num2str(i, '%03d'), '.png');
    imwrite(out,outfile);
end