% LINEDETECTORPOLAR.m - Line detector in Polar coordinate system
% 
% This code implements a line detector using Polar coordinate system. It
% iterate over each pixel and when an edge is detected, it will try to
% see the prospective lines depending on radius and the angle.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com

clear;clc;

inputimage = double(rgb2gray(imread('edges.png')));
%image size
[rows,columns]=size(inputimage);
%accumulator
rmax=round(sqrt(rows^2+columns^2));
acc=zeros(rmax,360);
%image
for x=1:columns
    for y=1:rows
        if(inputimage(y,x)==0)
            
            for m=1:360
                
                r=round(x*cos((m*pi)/180)+y*sin((m*pi)/180));
                
                if(r<rmax && r>0)
                    acc(r,m)= acc(r,m)+1;
                end
            end
        end
    end
end

j=max(max(acc));
[d,e]=find(acc==j); % This will retrieve where is the maxima
theta = e(round(size(e,1)/2));
r = d(round(size(d,1)/2));

figure;imshow(inputimage);
hold on;

% Drawing the line
thetaComp = 90 - theta;
point = round([r*cos(thetaComp*pi/180) r*sin(thetaComp*pi/180)]);

x = [0:.1:1000];
m = tan(thetaComp*pi/180);
y = -m*(x-point(2))+point(1);
    
plot(x,y);
