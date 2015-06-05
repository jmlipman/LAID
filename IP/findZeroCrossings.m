% FINDZEROCROSSINGS.m - Find zero-crossings
% 
% This function returns a matrix where zero-crossings are produced. It
% receives the image where it will try to find zero-crossings.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param image: matrix where zero-crossing will be performed.
% @param preserve: if the value is 0, the size of the matrix will not be
% preserved, and therefore, it will be smaller depending on the size of the
% template.
% @return result: matrix after the zero-crossing.

function [ result ] = findZeroCrossings( image, preserve )

    whiteColor = 255;
    blackColor = 0;

    % This code here is basically the same as in myConv.m
    % It will iterate over the whole picture.
    templateHeight = 3;
    templateWidth = 3;

    dataHeight = size(image,1);
    dataWidht = size(image,2);

    verticalSeparation = floor(templateHeight/2);
    horizontalSeparation = floor(templateWidth/2);

    if preserve==0
        offsetX=horizontalSeparation;
        offsetY=verticalSeparation;
    else
        offsetX=0;
        offsetY=0;
        result = zeros(dataHeight,dataWidht);
    end

    % Walk through the data to do the convolution
    for y=1+verticalSeparation:dataHeight-verticalSeparation
       fprintf('%i/%i\n',y,dataHeight-verticalSeparation);
       for x=1+horizontalSeparation:dataWidht-horizontalSeparation 
           winInitY = y-verticalSeparation;
           winInitX = x-horizontalSeparation;
           window = double(image(winInitY:winInitY+templateHeight-1,winInitX:winInitX+templateWidth-1));

           % It will calculate whether there is a zero-crossing by detecting
           % differences between 4 sections in a 3x3 window
           sections = [mean([window(1,1) window(1,2) window(2,1) window(2,2)])
               mean([window(1,2) window(2,2) window(1,3) window(2,3)])
               mean([window(2,2) window(2,3) window(3,2) window(3,3)])
               mean([window(2,1) window(2,2) window(3,1) window(3,2)])];

           if max(sections)>=0 && min(sections)<0
              result(y-offsetY,x-offsetX) = whiteColor;
           else
               result(y-offsetY,x-offsetX) = blackColor;
           end
       end
    end

end

