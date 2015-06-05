% MARR_HILDRETH.m - Marr-Hildreth operator example
% 
% This code implements Marr-Hildreth operator. It uses the second
% derivative of gaussian to create the template to convolute later, and
% finally, it detects zero-crossings to establish whether an edge was
% found or not.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com

clear;clc;

% Whether we preserve the real size of the image or we cut it.
% When the image is convoluted with the template, the borders are always
% lost.
preserve = 1;


% Read the image in gray scale
im = rgb2gray(imread('utebo.jpg'));

% Calculate a matrix which will be the template to convolute. Sigma and
% size of the kernel are given to the function.
gauss = secDerGauss(0.7,9);

% Normalization
% The trials I carried on showed me that there was no difference between
% normalizing and not.
%ratio = 1/(sum(sum(gauss)));
%gauss = gauss.*ratio;

% Convolution through the whole image
convoluted = myConv( im, gauss, preserve);

% Find zero-crossings
result = findZeroCrossings(convoluted, preserve);

imshow(result);
