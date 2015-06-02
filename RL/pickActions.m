% PICKACTIONS.m - Pick actions given the current state.
% 
% This function will return a matrix with the actions it can do given
% a certain maze configuration and state.
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
% @param V: matrix containing V values.
% @return targetStates: set of states or actions (with its corresponding 
% V value) that can be done from the current state.


function [ targetStates ] = pickActions( currentState, hMdt, hMlr, V )
    
    [row,column] = state2cells(currentState);
    
    hMdt = flipud(hMdt);
    hMlr = flipud(hMlr);
    targetStates = [];
    % Check 4 sides
    % Up
    if row>1
        r = row-1;
        c = column;
        if ~hMdt(r,c)
            targetStates = [targetStates; 6*(r-1)+c V(r,c)];
        end
    end
    
    % Down
    if row<6
        r = row;
        c = column;
        if ~hMdt(r,c)
            targetStates = [targetStates; 6*r+c V(r+1,c)];
        end
    end
    
    % Left
    if column>1
        r = row;
        c = column-1;
        if ~hMlr(r,c)
            targetStates = [targetStates; 6*(r-1)+c V(r,c)];
        end
    end
    
     % Right
    if column<6
        r = row;
        c = column;
        if ~hMlr(r,c)
            targetStates = [targetStates; 6*(r-1)+c+1 V(r,c+1)];
        end
    end
        
    % Sorted by V(s)
    targetStates = sortrows(targetStates,2);
    
end

