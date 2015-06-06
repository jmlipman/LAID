% SOBELFILTER.m - Sobel Filter
% 
% This function implements Sobel filter. It only needs an image to apply
% the filter, but it can also calculate the direction of the energy used
% by Canny detector.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param image: image to which the filter is applied.
% @param (optional) saveDirections: if it is 1, it will save directions
% (for Canny)
% @return result: matrix after the convolution.
% @return (optional) directions: directions obtained for Canny.

function [ result, directions ] = sobelFilter(image, saveDirections)

    switch nargin
        case 1
            saveDirections = 0;
            directions = 0;
    end

    preserve = 1;
    
    templateX = [-1 0 1;
        -2 0 2;
        -1 0 1;];
    
    templateY = [1 2 1;
        0 0 0;
        -1 -2 -1;];
    
    Gx = myConv(image, templateX, preserve);
    Gy = myConv(image, templateY, preserve);
    result = uint8(sqrt(Gx.^2+Gy.^2));
    
    if saveDirections==1
        height = size(result,1);
        width = size(result,2);
        directions = zeros(height,width);
        
       for y=1:height
           for x=1:width
               
               angle = atan2(Gy(y,x),Gx(y,x))
               
               if angle>=22.5 && angle <=67.5
                   % Vertical-right /
                   directions(y,x) = 3;
               elseif angle>67.5 && angle<=112.5
                   % Vertical |
                   directions(y,x) = 1;
               elseif angle>112.5 && angle<=157.5
                   % Vertical-left \
                   directions(y,x) = 4;
               else 
                   % Horizontal -
                   directions(y,x) = 2;
               end
               
           end
       end
        
    end
    
end

