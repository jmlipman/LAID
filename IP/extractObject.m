% EXTRACTOBJECT.m - Extract object from background
% 
% This function subtracts one picture from another. If the difference is
% greater than a certain threshold, that pixel will be colored. Otherwise,
% the color will be the a different background color.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param backgroundIm: background image.
% @param figureIm: image with the figure we want to extract.
% @param (optional) threshold: threshold to consider changes in the
% pictures.
% @param (optional) grayscale: whether to use grayscale or not.
% @return result: extracted feature.

function [ result ] = extractObject( backgroundIm, figureIm, threshold, grayscale)


    switch nargin
        case 2
            threshold = 40;
            grayscale = 0;
        case 3
            grayscale = 0;
    end

    % Feature color (green)
    redColorFigure = 10;
    greenColorFigure = 240;
    blueColorFigure = 20;

    % Background color (black)
    redColorBackground = 0;
    greenColorBackground = 0;
    blueColorBackground = 0;

    % Gaussian
%     gaussTemplate = fspecial('gaussian', 9, 1);
%     backgroundIm = myConv(backgroundIm, gaussTemplate, 1);
%     figureIm = myConv(backgroundIm, gaussTemplate, 1);

    height = size(backgroundIm,1);
    width = size(backgroundIm,2);

    % RGB. Background by default
    result(:,:,1) = redColorBackground*ones(height,width);
    result(:,:,2) = greenColorBackground*ones(height,width);
    result(:,:,3) = blueColorBackground*ones(height,width);

    % Finally we compare them. If pixels are different, pixels that belongs
    % to the image we want to extract features will be used.
    
    backgroundIm = double(backgroundIm);
    figureIm = double(figureIm);

    if grayscale==1
        for y=1:height
            for x=1:width

                if backgroundIm(y,x)-figureIm(y,x)>threshold
                    result(y,x,1) = redColorFigure;
                    result(y,x,2) = greenColorFigure;
                    result(y,x,3) = blueColorFigure;
                end

            end
        end  
    else
        for y=1:height
            for x=1:width

%                  if (backgroundIm(y,x,1)-figureIm(y,x,1)>threshold && ...
%                      backgroundIm(y,x,2)-figureIm(y,x,2)>threshold && ...
%                      backgroundIm(y,x,3)-figureIm(y,x,3)>threshold)
                dif1 = abs(backgroundIm(y,x,1)-figureIm(y,x,1));
                dif2 = abs(backgroundIm(y,x,2)-figureIm(y,x,2));
                dif3 = abs(backgroundIm(y,x,3)-figureIm(y,x,3));
                 %if (dif1+dif2+dif3>threshold)
                 
                 if (dif1>threshold || dif2>threshold || dif3>threshold)
                    result(y,x,1) = redColorFigure;
                    result(y,x,2) = greenColorFigure;
                    result(y,x,3) = blueColorFigure;
                end

                
        end  
    end

    result = uint8(result);

end

