% MAZE1FVMC.m - Solving a Maze (first-visit Monte Carlo).
% 
% This code is a graphical representation of a 300x300 px maze that has 36 
% states (6x6 matrix) solved by an implementation of first-visit Monte
% Carlo algorithm. The left side of the representation is the maze itself,
% which will be colored by the states the algorithm visited. The right side
% is the same maze with its V values, which represented the amount of steps
% to be taken from each cell to reach the goal.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%


clear;clc;

%% First part, drawing the GUI. This slows down the code quite a lot.

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
             
annotation('textbox', [0.6 0.89 0.1 0.1],...
                'fontsize', 12,'linestyle', 'none',...
                'string', 'Iterations: ',...
                 'color', [0 0.5 0]);
             
iter_h = annotation('textbox', [0.72 0.89 0.1 0.1],...
                'fontsize', 12,'linestyle', 'none',...
                'string', 0,...
                 'color', [0 0.5 0]);
            
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

% Vertical
% From left to right
hMlr = [0 0 0 1 0;
    1 0 0 1 1;
    1 0 0 0 0;
    0 0 0 1 0;
    1 1 1 1 0;
    0 0 1 0 0;];

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



%% Second part, the algorithm

% How likely is to choose a random action
eps = 0.1;
% In the begining, V values are the amount of states (36) because it cannot
% take longer than 36 steps. This way, it will always try to find a
% smaller (better) value.
V = (6*6)*ones(6,6);

% Counters
counter = 1;
maxCounter = 5; % max of iters
finalSum = 0;
% This will store the amount of iterations per episode
totalIterations = [];
while counter~=(maxCounter-1)
    
    % It always start in the state 1
    episode = [1];
    counterIter = 0;
    
    % I will get elements til I arrive to the last cell
    while episode(end)~=36 %last state, goal
        
        % Retrieval of all possible actions that can be done from the
        % current state
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
    
    % I delete the colored cells if any
    delete(findall(gcf, 'tag', 'tmpcell'));
    % I delete the current V(s) values
    delete(findall(gcf, 'tag', 'tmptext'));
    % I draw the new cells
    colorCellsA(episode);
    
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
    
    % I update the V(s) values
    updateVfield(V);
    
    % Increment the amount of iterations
    set(iter_h,'string',counter);
    
    
    pause(1);
    
    
end



