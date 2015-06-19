% LINEDETECTORCARTESIANIMP.m - Line detector in Cartesian coordinate system
% 
% This code implements a line detector using Cartesian coordinate system.
% This improved version works for all cases. 
% It iterates over each pixel and determine all its
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
acc1=zeros(rows,181);
acc2=zeros(columns,181);

% Walk through the image
for x=1:columns
    for y=1:rows
        if(inputimage(y,x)==0)
            for m=-45:135
                % change the coordinates system
                yInv = rows-y;
                
                b=round(yInv-tan((m*pi)/180)*x);
                
                if(b<rows && b>0)
                    acc1(b,m+45+1)= acc1(b,m+45+1)+1;
                end
                
                % intersection in x-axis
                b=round(x-yInv/tan((m*pi)/180));
                
                if(b<columns && b>0)
                    acc2(b,m+45+1)= acc2(b,m+45+1)+1;
                end
                
            end
            
         end
     end
 end


% Calculate the maxima to draw the line whose characteristics are found in
% the coordinates of that maxima
j1=max(max(acc1));
j2=max(max(acc2));
if j1>j2
    [d,e]=find(acc1==j1);
    e = e(round(size(e,1)/2));
    d = d(round(size(d,1)/2));
    e = (e-46);
    point = round([rows-d 0]);
else
    [d,e]=find(acc2==j2);
    e = e(round(size(e,1)/2));
    d = d(round(size(d,1)/2));
    e = (e-46);
    point = round([rows d]);
end



figure;imshow(inputimage);
hold on;

% Drawing the line

theta = e;


x = [0:.1:1000];
m = tan(theta*pi/180);
y = -m*(x-point(2))+point(1);
    
    
plot(x,y);
