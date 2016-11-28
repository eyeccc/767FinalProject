function out = preprocess(A)
  A = imcomplement(A);
  B = medfilt2(A);
  B = highboost(B);
  %imshow(B);
  %B = imadjust(B,stretchlim(B),[]);
  J = histeq(B);
  %imshow(J);
  J = medfilt2(J);
  %imshow(J);
  J = imadjust(J,stretchlim(J),[]);
  J = imcomplement(J);
  %imshow(J);
  out = J;%im2bw(J, graythresh(J));
end

function o = highboost(img)
  m = [1 1 1; 1 1 1; 1 1 1]/9;
  f = imfilter(img, m);
  k = 4;
  u = [-1 -1 -1;-1 k+4 -1; -1 -1 -1];
  fh = imfilter(img,u) + f;
  o = fh;
end