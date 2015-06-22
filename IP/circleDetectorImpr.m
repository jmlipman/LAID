% CIRCLEDETECTORIMPR.m - Circle detector after Space Reduction
% 
% This code is an improvement of the Circle Detector algorithm when Space
% Reduction is applied.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com

clear;clc;

im = rgb2gray(imread('circle.png'));
[rows, columns] = size(im);
acc = zeros(rows,columns);
rad = zeros(1,round(sqrt(2)*max([rows columns])));

for x1=1:columns
    for y1=1:rows
        if im(y1,x1)==0
            % Analyze the neighborhood between 12 and 10 pixels.
            % I think this could be improved
            for x2=x1-12:x1+12
                for y2=y1-12:y1+12
                    if (abs(x2-x1)>10 || abs(y2-y1)>10)
                        if (x2>0 && y2>0 && x2<columns && y2<rows)
                            % When a pixel is detected in the neighborhood
                            if(im(y2,x2)==0)
                                xm = (x1+x2)/2;
                                ym = (y1+y2)/2;
                                % Calculate the slope
                                if (y2-y1~=0)
                                    m=((x2-x1)/(y2-y1));
                                else
                                    m=99999999;
                                end
                                % Calculate the corresponding line to be
                                % added in the accumulator
                                if m>-1 && m<1
                                    for x0=1:columns
                                        y0 = round(ym+m*(xm-x0));
                                        
                                        if (y0>0 && y0<rows)
                                            acc(y0,x0)=acc(y0,x0)+1;
                                        end
                                    end
                                else
                                    for y0=1:rows
                                        x0 = round(xm+(ym-y0)/m);
                                        if (x0>0 && x0<columns)
                                            acc(y0,x0)=acc(y0,x0)+1;
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

[y,x]=find(acc==max(max(acc)))

% I take the value in the middle, in case I get more than 1 value
y = y(round(length(y)/2));
x = x(round(length(x)/2));

% To calculate the radius, we calculate the distance from each pixel to the
% center previusly obtained. This strategy is invalid when faced against
% more than one circles to be detected.
for x1=1:columns
    for y1=1:rows
        if im(y1,x1)==0
            d = round(sqrt((x-x1)^2+(y-y1)^2));
            rad(d) = rad(d)+1;
        end
    end
end

[dummy,r] = max(rad);

figure, imshow(im);

% Draw over the picture the generated circle
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
plot(xunit, yunit);
