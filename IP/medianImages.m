% MEDIANIMAGES.m - Sobel Filter
% 
% This code shows how temporal median works. It takes different pictures
% taken from the same spot during different times and obtains the median
% from the same coordinates through the different images.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com

clear;clc;

pic1_ = imread('1.jpg');
pic2_ = imread('2.jpg');
pic3_ = imread('3.jpg');
pic4_ = imread('4.jpg');

height = size(pic1_,1);
width = size(pic1_,2);

for y=1:height
   for x=1:width
        
       red = double([pic1_(y,x,1) pic2_(y,x,1) pic3_(y,x,1) pic4_(y,x,1)]);
       green = double([pic1_(y,x,2) pic2_(y,x,2) pic3_(y,x,2) pic4_(y,x,2)]);
       blue = double([pic1_(y,x,3) pic2_(y,x,3) pic3_(y,x,3) pic4_(y,x,3)]);
    
       result(y,x,1) = uint8(median(red));
       result(y,x,2) = uint8(median(green));
       result(y,x,3) = uint8(median(blue));
       
   end
   %fprintf('%i/%i\n',y,height);
end

imshow(result)
