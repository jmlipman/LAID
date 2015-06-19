% LINEDETECTORCARTESIAN.m - Line detector in Cartesian coordinate system
% 
% This code implements a line detector using Cartesian coordinate system.
% This version does not work for all cases, but as it is simple, it is
% easier to understand. It iterates over each pixel and determine all its
% prospective lines. The final line will be the one found in the
% accumulator with the maximum value.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com

clear;clc;

inputimage = double(rgb2gray(imread('edges2.png')));
%image size
[rows,columns]=size(inputimage);
%accumulator
acc1=zeros(rows,91);
acc2=zeros(columns,91);

%image
for x=1:columns
    for y=1:rows
        if(inputimage(y,x)==0)
            % For horizontal crossings
            for m=-45:45
                b=round(y-tan((m*pi)/180)*x);
                if(b<rows && b>0)
                    acc1(b,m+45+1)= acc1(b,m+45+1)+1;
                end
            end
            % For vertical crossings
            for m=45:135
                b=round(x-y/tan((m*pi)/180));
                if(b<columns && b>0)
                    acc2(b,m-45+1)= acc2(b,m-45+1)+1;
                end
            end
         end
     end
end

 % Depending on which accumulator has the larger value, the line will be
 % drawn from the x-axis or the y-axis
 j1=max(max(acc1));
 j2=max(max(acc2));
 if j1>j2
    [d,e]=find(acc1==j1);
    e = e(round(size(e,1)/2));
    d = d(round(size(d,1)/2));
    e = 1*(2*e-46);
    point = round([rows d]);
 else
    [d,e]=find(acc2==j2)
    e = e(round(size(e,1)/2));
    d = d(round(size(d,1)/2));
    e = -1*(e+44);
    point = round([0 d]);
 end



figure;imshow(inputimage);
hold on;

% Drawing of the line
theta = e;

x = [0:.1:1000];
m = tan(theta*pi/180);
y = -m*(x-point(2))+point(1);
    
plot(x,y);
