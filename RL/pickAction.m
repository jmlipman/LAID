% PICKACTION.m - Pick an action given the current state.
% 
% This function will return an action decided to be taken.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param currentState: current state from which actions will be explored
% and chosen.
% @param hMdt: horizontal maze walls, part of the maze's configuration.
% @param hMlr: vertical maze walls, part of the maze's configuration.
% @param Q: matrix containing Q values.
% @param eps: epsilon value, how likely it is to choose a random option.
% @return selectedAction: action (and its corresponding Q value) that has
% been selected.
% @return QValue: qvalue

function [ selectedAction QValue ] = pickAction( currentState, hMdt, hMlr, Q, eps )
    % targetStates = [s' a S(s',a)]
    
    % How likely is to choose a random action
    
    
    [row,column] = state2cells(currentState);
    
    hMdt = flipud(hMdt);
    hMlr = flipud(hMlr);
    actions = [];
    % Check 4 sides
    % Up
    if row>1
        r = row-1;
        c = column;
        if ~hMdt(r,c)
            actions = [actions; 1 Q(currentState,1)];
        end
    end
    
    % Down
    if row<6
        r = row;
        c = column;
        if ~hMdt(r,c)
            actions = [actions; 3 Q(currentState,3)];
        end
    end
    
    % Left
    if column>1
        r = row;
        c = column-1;
        if ~hMlr(r,c)
            actions = [actions; 4 Q(currentState,4)];
        end
    end
    
     % Right
    if column<6
        r = row;
        c = column;
        if ~hMlr(r,c)
            actions = [actions; 2 Q(currentState,2)];
        end
    end
        
    % Sorted by S(s,a)
    actions = flipud(sortrows(actions,2));
    
    %disp(actions);
    % This will calculate how many best options I have
    totalBest = length(find(actions(:,2)==max(actions(:,2))));
    %fprintf('totalBest: %i\n',totalBest);
    if rand > eps % I will usually use the best state
     % Choose a random state among the best options
     
        randomIndex = randi([1 totalBest],1,1);

    else % Ocasionally, I will choose a random non-optimal state
        % I need to take care about the size of the matrix
        if size(actions,1)==1
            initIndex = 1;
        else
            initIndex = 2;
        end
        randomIndex = randi([initIndex size(actions,1)],1,1);
    end
    
    selectedAction = actions(randomIndex,1);
    QValue = actions(randomIndex,2);
    
end

