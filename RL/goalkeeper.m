% GOALKEEPER.m - Eager goalkeeper
% 
% This code is a graphical representation of a goalkeeper trying to block
% every ball during an infinite loop. This algorithm is 100% explorative,
% so between a good option and an unknown option, it will always choose the
% unknown. After that, it will learn if it was a good or a bad option.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%

clear;clc;

totalPositions = 5;
fieldRelativeHeight = .6;
playerHeight = fieldRelativeHeight/totalPositions;

speed = .01;

% Position of the player and the ball
positionPlayer = 0;
positionBall = 0;
counterScored = 0;
counterBlocked = 0;

% Matrix which contains all possible moves it can do
moves = -99*ones(totalPositions^2);

moveMatrixIndex=1;
for a=0:-1:-(totalPositions-1)
    for b=1:totalPositions
       for c=1:totalPositions
          moves(moveMatrixIndex,b+(c-1)*totalPositions)=a+(c-1);
       end
       moveMatrixIndex= moveMatrixIndex+1;
    end
end




% If the reward is -99, then it's not possible to do that move.
% If the reward is -1, then it is a bad option.
% If the reward is 1, then it is a good option which will lead to blocking
% the ball.
% If the reward is 2, then it is an unkonwn option and therefore it does
% not known whether is good or bad, but it will choose it since it is an
% exporative algorithm.
RM = -101*(moves==-99);
RM = RM + 2;


figure('resize', 'off', 'position', [400 150 700 450]);

% Field
field_h = annotation('rectangle', [.05,.3,.9,fieldRelativeHeight],...
                'color', [0 0 0], 'facecolor', 'white');
            
% Player     
player_h = annotation('rectangle', [.05,.3+playerHeight*positionPlayer,.02,playerHeight],...
                'color', [0 0 0], 'facecolor', 'black');

% Ball
ballPositionReset = [.9,.4+.2*positionBall,.04,.05];
ball_h = annotation('ellipse', ballPositionReset,...
                'color', [1 1 1], 'facecolor', 'white');         

% Labels

annotation('textbox', [0.1 0.89 0.1 0.1],...
                'fontsize', 12,'linestyle', 'none',...
                'string', 'Author: Juan Miguel Valverde Martinez',...
                 'color', [0.3 0.3 0.3]);
            
            
annotation('textbox', [0.1 0.15 0.1 0.1],...
                'fontsize', 12, 'string', 'Scored:',...
                'linestyle', 'none', 'color', [1 0 0]);
            
scoredCounter_h = annotation('textbox', [0.2 0.15 0.1 0.1],...
                'fontsize', 12, 'string', 0,...
                'linestyle', 'none', 'color', [1 0 0]);
            
annotation('textbox', [0.5 0.15 0.1 0.1],...
                'fontsize', 12, 'string', 'Blocked:',...
                'linestyle', 'none', 'color', [0 0.5 0]);
            
blockedCounter_h = annotation('textbox', [0.61 0.15 0.1 0.1],...
                'fontsize', 12, 'string', 0,...
                'linestyle', 'none', 'color', [0 0.5 0]);
            
            
%totalPositions
positionBall = randi([0 totalPositions-1],1,1);
ballIncrement = playerHeight/2;
set(ball_h, 'Position', [.9,.3+ballIncrement+2*ballIncrement*positionBall,.04,.05],...
    'color', [0 0 0], 'facecolor', 'black');


currentState = 1+totalPositions*positionPlayer+positionBall;

while 1==1
    % Evaluation function
    eval = 0;    
    
    % Let's move the ball or remove it if it reached the limit
    ballCurrentPosition = get(ball_h,'Position');
    % X axis
    if ballCurrentPosition(1)<0.06
        
        if positionBall == positionPlayer
            counterBlocked = counterBlocked + 1;
            eval = 1;
            
        else
            counterScored = counterScored + 1;
            eval = -1;
            
        end
        
        set(scoredCounter_h, 'string', counterScored);
        set(blockedCounter_h, 'string', counterBlocked);
        
        
        previousState = 1+totalPositions*positionPlayer+positionBall;
        
        positionBall = randi([0 totalPositions-1],1,1);
        set(ball_h, 'Position', [.9,.3+ballIncrement+2*ballIncrement*positionBall,.04,.05],...
            'color', [0 0 0], 'facecolor', 'black');
        
        RM(currentState,previousState) = eval;
        
        currentState = 1+totalPositions*positionPlayer+positionBall;
        
        % This can be problematic to understand. I'm actually learning it
        % in such way that, if this matrix is M(A,B), A will be the origin
        % state and B will be the target state. The value will be the
        % evaluation function.
        % currentState variable actually stores the value that will become
        % the previous state.
        
        
        % This will return which movement is the best
        [whatever,targetState]=max(RM(currentState,:));

        % It can be calculated the distance from the one state to another
        distance = moves(currentState,targetState)
        
        % This function moves the player to wherever it should be
        positionPlayer = move( player_h, positionPlayer, distance, playerHeight, totalPositions );
        
    else
        ballCurrentPosition(1) = ballCurrentPosition(1)-0.02;
        set(ball_h, 'Position', ballCurrentPosition);
    end
    
    
    pause(speed);
end





