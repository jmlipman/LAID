% REMOVEBACKGROUND.m - Code to remove background
% 
% This script implements a simple thresholding and subtraction to show the
% amazing results we can get by just these two really simple operations. At
% first, this algorithm takes a picture of the background and waits 4-5
% seconds. Later, it takes a certain amount of pictures while performing
% thresholding and subtraction with respect to the background. The result,
% is that it detects new objects. The disadvantage of this extremely simple
% code is that is very sensitive to noise and luminosity changes.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com

clear;clc;

% First we preprocess both images
max_times = 50;

% Configure webcam. This varies depending on your webcam.
% It is explained in the following post:
% http://laid.delanover.com/how-to-take-pictures-from-the-webcam-with-matlab/
vid = videoinput('winvideo', 1, 'MJPG_640x480');
vid.FramesPerTrigger = 1;  % 1 frame per second
vid.ReturnedColorspace = 'rgb'; % rgb channel

% Wait approximately 3 seconds
% (It takes 1.1 seconds to take a picture)
disp('Recording background in 2 seconds');
pause(1.9);
start(vid);
pause(1.1);

% Take picture of the background
backgroundPicture = getdata(vid);

% Wait 5 seconds
pause(3.9);
disp('Start moving!');
% Take picture every 1-2 seconds for max_times
while (max_times >= 0)
    fprintf('%i\n',max_times);
    max_times=max_times-1;
    start(vid)
    picture = getdata(vid);
    % Extract the differences
    result = extractObject(backgroundPicture, picture, 40, 0);
    imshow(result);
end


