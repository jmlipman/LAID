% LAPLACIAN.m - Laplacian operator example
% 
% This code implements Laplacian operator.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com

clear;clc;

preserve = 1;

template = [0 -1 0;
    -1 4 -1;
    0 -1 0;];

% Read image
im = rgb2gray(imread('utebo.jpg'));
% Convolution
convoluted = myConv( im, template, preserve);
% In some trials I found that it works better without doing the
% findZeroCrossings.
result = findZeroCrossings(convoluted, preserve);

imshow(result);

