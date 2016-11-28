function out = preprocess(A)
  A = imcomplement(A);
  B = medfilt2(A);
 % B = imadjust(B,stretchlim(B),[]);
  B = highboost(B);
  %imshow(B);
  
  %J = histeq(B);
  J = adapthisteq(B);
  %imshow(J);
  J = medfilt2(J);
  %imshow(J);
  J = imadjust(J,stretchlim(J),[]);
  %J = highboost(J);
  J = imcomplement(J);
  % R = imsharpen(J);
  %imshow(R);
  out = J;%im2bw(J, graythresh(J));
end

function o = highboost(img)
 % m = [1 1 1; 1 1 1; 1 1 1]/9;
  %f = imfilter(img, m);
  k = 4;
  u = [0 -k 0;-k 4*k+1 -k; 0 -k 0];
  fh = imfilter(img,u,'same');% + f;
  o = fh;
end