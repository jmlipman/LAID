% SECDERGASS.m - Second derivative of Gaussian
% 
% This function returns the calculated matrix with the values of the second
% derivative of the Gaussian function.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param sigma: sigma value.
% @param size: size of the final matrix.
% @return matrix: second derivate gaussian matrix.

function [ matrix ] = secDerGauss( sigma, size )

    if mod(size,2)==0 || size<3
        error('The size of the Guassian should be odd and greater or equal than 3.');
    end
    
    maxCoor = floor(size/2);
    addition = maxCoor+1;
    matrix = zeros(size);
    
    for x=0:maxCoor
        for y=0:maxCoor
            % Calculation of the value
            g = (1/sigma^2) * (((x^2+y^2)/sigma^2) - 2) * exp((-(x^2+y^2))/(2*sigma^2));
            
            matrix(y+addition,x+addition) = g;
            matrix(-y+addition,x+addition) = g;
            matrix(y+addition,-x+addition) = g;
            matrix(-y+addition,-x+addition) = g;
            
            matrix(x+addition,y+addition) = g;
            matrix(-x+addition,y+addition) = g;
            matrix(x+addition,-y+addition) = g;
            matrix(-x+addition,-y+addition) = g;
            
        end 
    end

end

