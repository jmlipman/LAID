% MYCONV.m - Convolution
% 
% This function returns the customized result of a convolution between a
% matrix an another. Given matrix A (image) and matrix B (template), the
% result will be the application of convolution of B through the whole A.
% You can also choose to preserve the real size of the image or not.
% Normally, when convolution is applied, the borders of the image are lost
% because it is not possible to convolve there.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param data: matrix which usually corresponds to the image.
% @param template: matrix to convolve with data.
% @param preserve: if the value is 0, the size of the matrix will not be
% preserved, and therefore, it will be smaller depending on the size of the
% template.
% @return result: matrix after the convolution.

function [ result ] = myConv( data, template, preserve )
    
    templateHeight = size(template,1);
    templateWidth = size(template,2);

    if mod(templateWidth,2)==0 || mod(templateHeight,2)==0  
        error('The size of the template should be odd in both dimensions.');
    end
    
    if templateWidth==1 && templateHeight==1
        error('Both dimensions'' size cannot be 1')
    end
   
    dataHeight = size(data,1);
    dataWidht = size(data,2);
    
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
       for x=1+horizontalSeparation:dataWidht-horizontalSeparation 
          
           winInitY = y-verticalSeparation;
           winInitX = x-horizontalSeparation;
           window = double(data(winInitY:winInitY+templateHeight-1,winInitX:winInitX+templateWidth-1));
           result(y-offsetY,x-offsetX) = sum(sum(window.*template));
       end
    end

end

