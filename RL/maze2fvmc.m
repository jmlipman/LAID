% MAZE2FVMC.m - Solving a Maze (first-visit Monte Carlo).
% 
% This code is a graphical representation of a 300x300 px maze that has 36 
% states (6x6 matrix) solved by an implementation of first-visit Monte
% Carlo algorithm. The left side of the representation is the maze itself,
% which will be colored by the states the algorithm visited. The right side
% is the same maze with its V values, which represented the amount of steps
% to be taken from each cell to reach the goal.
%
% Version 2:
%   - GUI has been boosted.
%   - You can decide whether or not using GUI.
%   - You can more clearly modify maze settings and configuration.
%   - Solution as a set of states is also printed.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%

clear;clc;

% Whether or not using GUI
GUI = 1;
% Start and goal states
startState = 6;
goalState = 20;

% Maze walls
% A 0 means that there is no wall. The values are INVERTED (from down to
% top)
% Horizontal
% From down to top.
hMdt = [0 1 0 1 0 0;
    0 0 1 1 1 1;
    1 1 1 0 1 0;
    0 0 1 0 1 0;
    0 1 0 0 1 1;];


% Vertical
% From left to right
hMlr = [0 0 0 1 0;
    1 0 1 1 1;
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

% How likely is to choose a random action
eps = 0.1;
% In the begining, V values is the amount of states (36)
V = (6*6)*ones(6,6);

% Flag
finish = 0;
% Counters
counter = 1;
maxCounter = 5; % max of iters
finalSum = 0;
% This will store the amount of iterations per episode
totalIterations = [];
while counter~=(maxCounter-1)
    
    % It always start in the state 1
    episode = [startState];
    counterIter = 0;
    
    % I will be getting elements til I arrive to the last cell
    while episode(end)~=goalState %last state, goal
        
        % Retrieval of all possible actions that can be done
        targetStates = pickActions(episode(end), hMdt, hMlr, V);
        % Best V(s) value among my options
        leastVvalue = targetStates(1,2);
        % This will calculate how many best options I have
        totalBest = length(find(targetStates(:,2)==min(targetStates(:,2))));
        
        if rand > eps % I will usually use the best state
            % Choose a random state among the best options
            randomIndex = randi([1 totalBest],1,1);
            
        else % Ocasionally, I will choose a random non-optimal state
            % I need to take care about the size of the matrix
            if size(targetStates,1)==1
                initIndex = 1;
            else
                initIndex = 2;
            end
            randomIndex = randi([initIndex size(targetStates,1)],1,1);
        end
        % Add a new state
        episode = [episode targetStates(randomIndex)];
        
        counterIter = counterIter + 1;
    end % I finally found the goal
    
    
    totalIterations = [totalIterations counterIter];
    counter = counter + 1;
    
    % Update values of V(s). I take the index of the last episode of each
    % state because it's the closest to the goal, and I keep its distance.
    episodeLength = length(episode);
    for a=1:36
        lastStateIndex = find(episode==a);
        if(length(lastStateIndex)>0)
           [row,column] = state2cells(a);
           
           prevValue = V(row,column);
           
           newValue = min([prevValue episodeLength-lastStateIndex(end)]);
           
           V(row,column) = newValue;
        end
    end
    
end

% Now I have create an episode with the optimal path. This is a copy of
% the previous code with a small difference. It is completely greedy
% since I just need to follow the best solution.

% I will be getting elements til I arrive to the last cell
finalEpisode = [startState];
while finalEpisode(end)~=goalState %last state, goal

    % Retrieval of all possible actions that can be done
    targetStates = pickActions(finalEpisode(end), hMdt, hMlr, V);
    % Best V(s) value among my options
    leastVvalue = targetStates(1,2);
    % This will calculate how many best options I have
    totalBest = length(find(targetStates(:,2)==min(targetStates(:,2))));

    index = randi([1 totalBest],1,1);

    % Add a new state
    finalEpisode = [finalEpisode targetStates(index)];

    counterIter = counterIter + 1;
end % I finally found the goal
    
disp('---Solution---');
disp(finalEpisode);

if GUI
     % The figure is displayed in the second screen
    figure('resize', 'off', 'position', [1600 50 625 375]);

    % Field
    field_h = annotation('rectangle');
    set(field_h, 'units', 'pixels', 'position', [12,20,300,300],...
        'color', [0 0 0], 'facecolor', 'white');

    % V(s) values
    notes_h = annotation('rectangle');
    set(notes_h, 'units', 'pixels', 'position', [319,20,300,300],...
        'color', [0 0 0], 'facecolor', 'white');

    annotation('textbox', [0.1 0.89 0.1 0.1],...
                    'fontsize', 12,'linestyle', 'none',...
                    'string', 'Author: Juan Miguel Valverde Martinez',...
                     'color', [0.3 0.3 0.3]);

    % Let's begin the maze construction
    % I know that the size of the maze is 300x300 px
    % Horizontal lines in both fields
    for a=0:1
        for b=1:5
            a_h = annotation('line');
            set(a_h, 'units', 'pixels', 'position',[12+a*307 20+50*b 300 0 ] );
        end
    end

    % Verrtical lines
    for a=0:1
        for b=1:5
            a_h = annotation('line');
            set(a_h, 'units', 'pixels', 'position',[12+b*50+a*307 20 0 300 ] );
        end
    end

    % Vertical perimeter
    a_h = annotation('line', 'LineWidth', 3);
    set(a_h, 'units', 'pixels', 'position',[12 20 0 300 ] );
    a_h = annotation('line', 'LineWidth', 3);
    set(a_h, 'units', 'pixels', 'position',[12+300 20 0 300 ] );

    % Horizontal perimeter
    a_h = annotation('line', 'LineWidth', 3);
    set(a_h, 'units', 'pixels', 'position',[12 20 300 0 ] );
    a_h = annotation('line', 'LineWidth', 3);
    set(a_h, 'units', 'pixels', 'position',[12 20+300 300 0 ] );

    a_ = size(hMdt,1); % height
    b_ = size(hMdt,2); % width

    % Drawing it
    for a=1:a_
       for b=1:b_
           if hMdt(a,b)~=0
             a_h = annotation('line', 'LineWidth', 3);
             set(a_h, 'units', 'pixels', 'position',[12+50*(b-1) 20+50*(a) 50 0 ] );
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
             set(a_h, 'units', 'pixels', 'position',[12+50*(b) 20+50*(a-1) 0 50 ] );
           end
       end
    end   
    

    
    colorCellsA(finalEpisode);
    updateVfield(V);
    
end
