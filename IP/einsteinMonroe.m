% EINSTEINMONROE.m - High and Low-pass filter examples in Image Processing
% 
% This code shows how features from images can be extracted using low and
% high pass filters. 
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com

clear;clc;

im = rgb2gray(imread('einsteinmonroe','png'));
% Show the original image
% imshow(im);
[rows,columns] = size(im);
% Perform 2-D Fast Fourier Transform
fouriered = fft2(double(im));
% Shift the result
shifted = fftshift(fouriered);
% Show the Fourier Result (no scaled)
%imshow(uint8(shifted));


tt = log(abs(shifted));
minN = min(min(tt));
maxN = max(max(tt));
tt = (tt-minN).*(255/(maxN-minN));
% Show the Fourier Result (scaled)
%imshow(uint8(tt));

% The filter is built using a zero-matrix which has 1s in the middle
% drawing a circle. When this matrix is multiplied by the Fast Fourier
% matrix, this will be finally filtered.

% Filter Radius
rad = 18;
[cc rr] = meshgrid(1:columns,1:rows);
Coriginal = sqrt((rr-rows/2).^2+(cc-columns/2).^2)<=rad;
C_0 = Coriginal==0; %1 -> Monroe, 0 -> Einstein
C_1 = Coriginal==1;

% Matrix before obtaining the High-pass filter
% imshow(C_0);
% Matrix before obtaining the Low-pass filter
% imshow(C_1);

z1 = shifted.*C_0;
z2 = shifted.*C_1;

% Matrix after obtaining the High-pass filter
%imshow(uint8(tt.*C_0));
% Matrix after obtaining the Low-pass filter
%imshow(uint8(tt.*C_1));

% Unshift and un-fourier
un1 = ifftshift(z1);
un2 = ifftshift(z2);
final1 = ifft2(un1);
final2 = ifft2(un2);
% High frequency image: Einstein
figure, imshow(uint8(final1));
% Low frequency image: Monroe
figure, imshow(uint8(final2));


% Increase the contrast of the High frequency image
image = final1;
[rows,columns] = size(image);

for a=1:rows
    for b=1:columns
        if image(a,b)>10
            image(a,b)=4*image(a,b);
        end
    end
end
% Show it
figure, imshow(uint8(image));
 
