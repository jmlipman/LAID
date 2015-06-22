% CIRCLEDETECTOR.m - Circle detector (Hough Transform)
% 
% This code is an implementation of the circle detector using a 3-D
% accumulator (x,y,radius).
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com

clear;clc;

im = rgb2gray(imread('circle.png'));
[rows, columns] = size(im);
% Range of radius to be analized
rstart = 10;
rmax = 30;
% Accumulator
acc = zeros(rows,columns,rmax-rstart);

for x=1:columns
    for y=1:rows
        if im(y,x)==0
            
            %fprintf('Point detected at [%i,%i]\n',y,x);
            % Iterate over the range of radius
            for r=1:rmax-rstart
                for ang=0:360
                    rad = r+rstart;
                    t = ang*pi/180;
                    % Generate the coordinates
                    x0 = round(x-rad*cos(t));
                    y0 = round(y-rad*sin(t));

                    if (x0<columns && x0>0 && y0<rows && y0>0)
                        acc(y0,x0,r) = acc(y0,x0,r)+1;
                    end
                end
            end
        end
    end
end

figure, imshow(im);
% Obtain the coordinates of the maximum value of the accumulator
[y,x,r]=ind2sub(size(acc),find(acc==max(max(max(acc)))));
r=r+rstart;

% Draw over the picture the generated circle
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
plot(xunit, yunit);
