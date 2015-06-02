% MAZESARSA.m - Sarsa solved maze
% 
% This code implements Sarsa algorithm in order to solve a 6x6 maze (36
% states). Features:
% - Activate or desactivate the GUI.
% - Auto generate a maze (walls) or leave it with no walls.
% - Choose learning rate, discounted reward and epsilon.
% - Choose maximum iterations and maximum itercoations within an episode.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%

clear;clc;

GUI = 1;
startState = 1;
% This always has to be 36. Otherwise, change calcReward function
goalState = 36;
% Maze walls
% A 0 means that there is no wall. The values are INVERTED (from down to
% top)
% Horizontal
% From down to top.
hMdt = [0 1 0 1 0 0;
    0 0 1 1 0 1;
    1 1 1 0 1 0;
    0 0 1 0 1 0;
    0 1 0 0 1 1;];


% Vertical
% From left to right
hMlr = [0 0 0 1 0;
    1 0 0 1 1;
    1 0 0 0 0;
    0 0 0 1 0;
    1 1 1 1 0;
    0 0 1 0 0;];

% Generate randomly the configuration of the maze
% This may make some errors because it can create walls and block all
% possible ways to achieve the goal.
% internalWalls needs to be between 0 and 1. If it is bigger than 1, it
% will preserve the previous configuration set below.
internalWalls = 2;
hMdt = rand(5,6) > internalWalls;
hMlr = rand(6,5) > internalWalls;



% S(s,a) values stand for all the states (36) up to 4 possible actions
% In the begining, all S values are zero
Q = zeros(36,4);
% Epsilon
eps = .1;
% Learning rate
alpha = 0.1;
% Discounted reward
gamma = 1;
% Flag
finish = 0;
% Counters
counter = 1;
% Max iterations within the same episode (to avoid getting stuck)
maxCounterIter = 1000;
% Max iterations
maxCounter = 300; 
% This will store the amount of iterations per episode
totalIterations = [];
while counter~=(maxCounter-1)
    
    % It usually starts in the state 1, but it can be changed
    currentState = startState;
    % I will take a random action
    currentAction = pickActionB(1, hMdt, hMlr, Q, eps);
    
    fprintf('Iteration: %i/%i\n',counter, maxCounter);
    counterIter = 0;
    
    % I will be getting elements til I arrive to the goal cell
    while currentState~=goalState %last state, goal
        
        % It can calculate the new state given current state and action
        newState = getNewState(currentState, currentAction);
        
        % Calculate a reward given current state and an action
        % This is done in an external function because it simulates the
        % unknown environment that we do not have access from the code
        reward = calcReward(currentState,currentAction);
        
        % Look forward to the next action using the same policy as before
        newAction = pickActionB(newState, hMdt, hMlr, Q, eps);

        % Sarsa update
        Q(currentState,currentAction) = Q(currentState,currentAction) + alpha*(reward + gamma*Q(newState,newAction) - Q(currentState,currentAction));
        
        counterIter = counterIter + 1;
        
        % New state
        currentState = newState;
        % New action
        currentAction = newAction;
        
        % To avoid getting stuck while iteration
        if counterIter > maxCounterIter
            break
        end
    end % I finally found the goal
    
    % This can be done to abort the algorithm when it gets stuck
    % if counter > 20 && counterIter > 100
    %    finish = 1;
    % end
    
    totalIterations = [totalIterations counterIter];
    counter = counter + 1;
   
end

% Now I have to create an episode with the optimal path. This is a copy of
% the previous code with a small difference. It is completely greedy
% since I just need to follow the best solution.

% I will be getting elements til I arrive to the last cell
finalEpisode = [startState];
while finalEpisode(end)~=goalState %last state, goal
    
    currentState = finalEpisode(end);
    % Retrieval of all possible actions that can be done
    action = pickActionB(currentState, hMdt, hMlr, Q, -1);
    state = getNewState(currentState, action);

    % Add a new state
    finalEpisode = [finalEpisode state];

    counterIter = counterIter + 1;
end % I finally found the goal
    
disp('---Parameters---');
fprintf('Alpha (learning rate): %.1s. Gamma (discounted reward): %d. Epsilon: %.1d.\n', alpha, gamma, eps);
disp('---Solution---');
disp(finalEpisode);

if GUI
     % The figure is displayed in the second screen
    figure('resize', 'off', 'position', [500 50 400 375]);

    % Field
    field_h = annotation('rectangle');
    set(field_h, 'units', 'pixels', 'position', [50,20,300,300],...
        'color', [0 0 0], 'facecolor', 'white');

    annotation('textbox', [0.1 0.89 0.1 0.1],...
                    'fontsize', 12,'linestyle', 'none',...
                    'string', 'Author: Juan Miguel Valverde Martinez',...
                     'color', [0.3 0.3 0.3]);

    % Let's begin the maze construction
    % I know that the size of the maze is 300x300 px
    % Horizontal lines in both fields
    for A=1:5
        a_h = annotation('line');
        set(a_h, 'units', 'pixels', 'position',[50 20+50*A 300 0 ] );
    end
    

    % Verrtical lines
    for A=1:5
        a_h = annotation('line');
        set(a_h, 'units', 'pixels', 'position',[50+A*50 20 0 300 ] );
    end
    

    % Vertical perimeter
    a_h = annotation('line', 'LineWidth', 3);
    set(a_h, 'units', 'pixels', 'position',[50 20 0 300 ] );
    a_h = annotation('line', 'LineWidth', 3);
    set(a_h, 'units', 'pixels', 'position',[50+300 20 0 300 ] );

    % Horizontal perimeter
    a_h = annotation('line', 'LineWidth', 3);
    set(a_h, 'units', 'pixels', 'position',[50 20 300 0 ] );
    a_h = annotation('line', 'LineWidth', 3);
    set(a_h, 'units', 'pixels', 'position',[50 20+300 300 0 ] );

    a_ = size(hMdt,1); % height
    b_ = size(hMdt,2); % width

    % Drawing it
    for a=1:a_
       for b=1:b_
           if hMdt(a,b)~=0
             a_h = annotation('line', 'LineWidth', 3);
             set(a_h, 'units', 'pixels', 'position',[50+50*(b-1) 20+50*(a) 50 0 ] );
           end
       end
    end

    a_ = size(hMlr,1); % height
    b_ = size(hMlr,2); % width

    % Drawing it
    for a=1:a_
       for b=1:b_
           if hMlr(a,b)~=0
             a_h = annotation('line', 'LineWidth', 3);
             set(a_h, 'units', 'pixels', 'position',[50+50*(b) 20+50*(a-1) 0 50 ] );
           end
       end
    end   
    
    colorCellsB(finalEpisode);
   
end
