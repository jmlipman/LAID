% CANNYEDGEDETECTOR.m - Canny Edge Detector
% 
% This function implements Canny edge detector steps in order to detect
% edges. This code is clearly divided into different steps in order to
% better understand the code. Additinally, many cases have been treated
% individually (corners and borders) to improve its performance.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param image: grayscale image to which the filter is applied. You can use
% rgb2gray(imread('image.jpg'))
% @param (optional) preserve: if the value is 0, the size of the matrix will not be
% preserved, and therefore, it will be smaller depending on the size of the
% template.
% @param (optional) gaussSize: size of the Gaussian applied.
% @param (optional) gaussSigma: sigma of the Gaussian applied.
% @return result: filtered image.

function [ result ] = cannyEdgeDetector( image, preserve, gaussSize, gaussSigma )
    
    switch nargin
        case 1
            preserve = 1;
            gaussSize = 9;
            gaussSigma = 1;
        case 2
            gaussSize = 9;
            gaussSigma = 1;
    end
    
    strongEdgeColor = 255;
    weakEdgeColor = 175;
    highThreshold = 50;
    lowThreshold = 20;
    
    gaussTemplate = fspecial('gaussian', gaussSize, gaussSigma);
    
    gauss = myConv(image, gaussTemplate, preserve);

    
    
    [sobel,directions] = sobelFilter(gauss,1);
    
    height = size(sobel,1);
    width = size(sobel,2);
    % -------------------->>>> Applying Non-Max Suppression
    maxSup = sobel;
    % First, we take care of the corners
    switch sobel(1,1)
        case 1 % vertical
            if sobel(1,1)<sobel(2,1)
                maxSup(1,1)=0;
            end
        case 2 % horizontal
            if sobel(1,1)<sobel(1,2)
                maxSup(1,1)=0;
            end
        case 4 % vertical-left \
            if sobel(1,1)<sobel(2,2)
                maxSup(1,1)=0;
            end
    end
    
    switch sobel(1,width)
        case 1 % vertical
            if sobel(1,width)<sobel(2,width)
                maxSup(1,width)=0;
            end
        case 2 % horizontal
            if sobel(1,width)<sobel(1,width-1)
                maxSup(1,width)=0;
            end
        case 3 % vertical-right /
            if sobel(1,width)<sobel(2,width-1)
                maxSup(1,width)=0;
            end
    end
    
    
    switch sobel(height,1)
        case 1 % vertical
            if sobel(height,1)<sobel(height-1,1)
                maxSup(height,1)=0;
            end
        case 2 % horizontal
            if sobel(height,1)<sobel(height,2)
                maxSup(height,1)=0;
            end
        case 3 % vertical-right /
            if sobel(height,1)<sobel(height-1,2)
                maxSup(height,1)=0;
            end
    end
    
    switch sobel(height,width)
        case 1 % vertical
            if sobel(height,width)<sobel(height-1,width)
                maxSup(height,width)=0;
            end
        case 2 % horizontal
            if sobel(height,width)<sobel(height,width-1)
                maxSup(height,width)=0;
            end
        case 4 % vertical-left \
            if sobel(height,width)<sobel(height-1,width-1)
                maxSup(height,width)=0;
            end
    end
     
        
    % Secondly, we take care of the borders
    % Vertical-left border:
    for y=2:height-1
       switch directions(y,1)
           case 1 % vertical
               if sobel(y,1)<sobel(y+1,1) || sobel(y,1)<sobel(y-1,1)
                  maxSup(y,1) = 0; 
               end
           case 2 % horizontal
               if sobel(y,1)<sobel(y,2)
                  maxSup(y,1)=0; 
               end
           case 3 % vertical-right /
               if sobel(y,1)<sobel(y-1,2)
                   maxSup(y,1)=0;
               end
           case 4 % vertical-left \
               if sobel(y,1)<sobel(y+1,2)
                  maxSup(y,1) = 0; 
               end
       end
       
    end
    
    % Vertical-right border:
    for y=2:height-1
       switch directions(y,width)
           case 1 % vertical
               if sobel(y,width)<sobel(y+1,width) || sobel(y,width)<sobel(y-1,width)
                  maxSup(y,width) = 0; 
               end
           case 2 % horizontal
               if sobel(y,width)<sobel(y,width-1)
                  maxSup(y,width)=0; 
               end
           case 3 % vertical-right /
               if sobel(y,width)<sobel(y+1,width-1)
                   maxSup(y,width)=0;
               end
           case 4 % vertical-left \
               if sobel(y,width)<sobel(y-1,width-1)
                  maxSup(y,width) = 0; 
               end
       end
    end
    
    % Horizontal-top border:
    for x=2:width-1
       switch directions(1,x)
           case 1 % vertical
               if sobel(1,x)<sobel(2,x)
                  maxSup(1,x) = 0; 
               end
           case 2 % horizontal
               if sobel(1,x)<sobel(1,x-1) || sobel(1,x)<sobel(1,x+1)
                  maxSup(1,x)=0; 
               end
           case 3 % vertical-right /
               if sobel(1,x)<sobel(2,x-1)
                   maxSup(1,x)=0;
               end
           case 4 % vertical-left \
               if sobel(1,x)<sobel(2,x+1)
                  maxSup(1,x) = 0; 
               end
       end
    end
    
    % Horizontal-bottom border:
    for x=2:width-1
       switch directions(height,x)
           case 1 % vertical
               if sobel(height,x)<sobel(height-1,x)
                  maxSup(height,x) = 0; 
               end
           case 2 % horizontal
               if sobel(height,x)<sobel(height,x-1) || sobel(height,x)<sobel(height,x+1)
                  maxSup(height,x)=0; 
               end
           case 3 % vertical-right /
               if sobel(height,x)<sobel(height-1,x+1)
                   maxSup(height,x)=0;
               end
           case 4 % vertical-left \
               if sobel(height,x)<sobel(height-1,x-1)
                  maxSup(height,x) = 0; 
               end
       end
    end
    
    % Within the picture with no 1-borders
    for y=2:height-1
       for x=2:width-1
          switch directions(height,x)
               case 1 % vertical
                   if sobel(y,x)<sobel(y-1,x) || sobel(y,x)<sobel(y+1,x)
                      maxSup(y,x) = 0; 
                   end
               case 2 % horizontal
                   if sobel(y,x)<sobel(y,x-1) || sobel(y,x)<sobel(y,x+1)
                      maxSup(y,x)=0; 
                   end
               case 3 % vertical-right /
                   if sobel(y,x)<sobel(y-1,x+1) || sobel(y,x)<sobel(y+1,x-1)
                      maxSup(y,x) = 0; 
                   end
               case 4 % vertical-left \
                   if sobel(y,x)<sobel(y-1,x-1) || sobel(y,x)<sobel(y+1,x+1)
                      maxSup(y,x) = 0; 
                   end
           end
       end
    end
    
    % -------------------->>>> End of Non-Max Suppression
    
    % -------------------->>>> Applying Hysteresis
    hyst = zeros(height,width);
    % First, we take care of the corners
    if maxSup(1,1)>=highThreshold
        hyst(1,1) = strongEdgeColor;
        if maxSup(1,2)>=lowThreshold && maxSup(1,2)<highThreshold
            hyst(1,2) = weakEdgeColor;
        end
        if maxSup(2,1)>=lowThreshold && maxSup(2,1)<highThreshold
            hyst(2,1) = weakEdgeColor;
        end
    end
    
    if maxSup(height,1)>=highThreshold
        hyst(height,1) = strongEdgeColor;
        if maxSup(height,2)>=lowThreshold && maxSup(height,2)<highThreshold
            hyst(height,2) = weakEdgeColor;
        end
        if maxSup(height-1,1)>=lowThreshold && maxSup(height-1,1)<highThreshold
            hyst(height-1,1) = weakEdgeColor;
        end
    end
    
    if maxSup(1,width)>=highThreshold
        hyst(1,width) = strongEdgeColor;
        if maxSup(1,width-1)>=lowThreshold && maxSup(1,width-1)<highThreshold
            hyst(1,width-1) = weakEdgeColor;
        end
        if maxSup(2,width)>=lowThreshold && maxSup(2,width)<highThreshold
            hyst(2,width) = weakEdgeColor;
        end
    end
    
    if maxSup(height,width)>=highThreshold
        hyst(height,width) = strongEdgeColor;
        if maxSup(height,width-1)>=lowThreshold && maxSup(height,width-1)<highThreshold
            hyst(height,width-1) = weakEdgeColor;
        end
        if maxSup(height-1,width)>=lowThreshold && maxSup(height-1,width)<highThreshold
            hyst(height-1,width) = weakEdgeColor;
        end
    end
    
    % Secondly, the borders
    % Vertical left
    for y=2:height-1
       if maxSup(y,1)>=highThreshold
          hyst(y,1)=strongEdgeColor;
          if maxSup(y-1,1)>=lowThreshold && maxSup(y-1,1)<highThreshold
              hyst(y-1,1)=weakEdgeColor;
          end
          if maxSup(y-1,2)>=lowThreshold && maxSup(y-1,2)<highThreshold
              hyst(y-1,2)=weakEdgeColor;
          end
          if maxSup(y,2)>=lowThreshold && maxSup(y,2)<highThreshold
              hyst(y,2)=weakEdgeColor;
          end
          if maxSup(y+1,2)>=lowThreshold && maxSup(y+1,2)<highThreshold
              hyst(y+1,2)=weakEdgeColor;
          end
          if maxSup(y+1,1)>=lowThreshold && maxSup(y+1,1)<highThreshold
              hyst(y+1,1)=weakEdgeColor;
          end
       end
    end
    
    % Vertical right
    for y=2:height-1
       if maxSup(y,width)>=highThreshold
          hyst(y,width)=strongEdgeColor;
          if maxSup(y-1,width)>=lowThreshold && maxSup(y-1,width)<highThreshold
              hyst(y-1,width)=weakEdgeColor;
          end
          if maxSup(y-1,width-1)>=lowThreshold && maxSup(y-1,width-1)<highThreshold
              hyst(y-1,width-1)=weakEdgeColor;
          end
          if maxSup(y,width-1)>=lowThreshold && maxSup(y,width-1)<highThreshold
              hyst(y,width-1)=weakEdgeColor;
          end
          if maxSup(y+1,width-1)>=lowThreshold && maxSup(y+1,width-1)<highThreshold
              hyst(y+1,width-1)=weakEdgeColor;
          end
          if maxSup(y+1,width)>=lowThreshold && maxSup(y+1,width)<highThreshold
              hyst(y+1,width)=weakEdgeColor;
          end
       end
    end
    
    % Horizontal top
    for x=2:width-1
       if maxSup(1,x)>=highThreshold
          hyst(1,x)=strongEdgeColor;
          if maxSup(1,x-1)>=lowThreshold && maxSup(1,x-1)<highThreshold
              hyst(1,x-1)=weakEdgeColor;
          end
          if maxSup(2,x-1)>=lowThreshold && maxSup(2,x-1)<highThreshold
              hyst(2,x-1)=weakEdgeColor;
          end
          if maxSup(2,x)>=lowThreshold && maxSup(2,x)<highThreshold
              hyst(2,x)=weakEdgeColor;
          end
          if maxSup(2,x+1)>=lowThreshold && maxSup(2,x+1)<highThreshold
              hyst(2,x+1)=weakEdgeColor;
          end
          if maxSup(1,x+1)>=lowThreshold && maxSup(1,x+1)<highThreshold
              hyst(1,x+1)=weakEdgeColor;
          end
       end
    end
    
    % Horizontal bottom
    for x=2:width-1
       if maxSup(height,x)>=highThreshold
          hyst(height,x)=strongEdgeColor;
          if maxSup(height,x-1)>=lowThreshold && maxSup(height,x-1)<highThreshold
              hyst(height,x-1)=weakEdgeColor;
          end
          if maxSup(height-1,x-1)>=lowThreshold && maxSup(height-1,x-1)<highThreshold
              hyst(height-1,x-1)=weakEdgeColor;
          end
          if maxSup(height-1,x)>=lowThreshold && maxSup(height-1,x)<highThreshold
              hyst(height-1,x)=weakEdgeColor;
          end
          if maxSup(height-1,x+1)>=lowThreshold && maxSup(height-1,x+1)<highThreshold
              hyst(height-1,x+1)=weakEdgeColor;
          end
          if maxSup(height,x+1)>=lowThreshold && maxSup(height,x+1)<highThreshold
              hyst(height,x+1)=weakEdgeColor;
          end
       end
    end
    
    % Finally, within the borders
    for y=2:height-1
        for x=2:width-1
            if maxSup(y,x)>highThreshold
                hyst(y,x)=strongEdgeColor;
                if maxSup(y-1,x-1)>=lowThreshold && maxSup(y-1,x-1)<highThreshold
                    hyst(y-1,x-1)=weakEdgeColor;
                end
                if maxSup(y-1,x)>=lowThreshold && maxSup(y-1,x)<highThreshold
                    hyst(y-1,x)=weakEdgeColor;
                end
                if maxSup(y-1,x+1)>=lowThreshold && maxSup(y-1,x+1)<highThreshold
                    hyst(y-1,x+1)=weakEdgeColor;
                end
                if maxSup(y,x-1)>=lowThreshold && maxSup(y,x-1)<highThreshold
                    hyst(y,x-1)=weakEdgeColor;
                end
                if maxSup(y,x)>=lowThreshold && maxSup(y,x)<highThreshold
                    hyst(y,x)=weakEdgeColor;
                end
                if maxSup(y,x+1)>=lowThreshold && maxSup(y,x+1)<highThreshold
                    hyst(y,x+1)=weakEdgeColor;
                end
                if maxSup(y+1,x-1)>=lowThreshold && maxSup(y+1,x-1)<highThreshold
                    hyst(y+1,x-1)=weakEdgeColor;
                end
                if maxSup(y+1,x)>=lowThreshold && maxSup(y+1,x)<highThreshold
                    hyst(y+1,x)=weakEdgeColor;
                end
                if maxSup(y+1,x+1)>=lowThreshold && maxSup(y+1,x+1)<highThreshold
                    hyst(y+1,x+1)=weakEdgeColor;
                end
                
            end
        end 
    end
    
    result = uint8(hyst);
    
end

