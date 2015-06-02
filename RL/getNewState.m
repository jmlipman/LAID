% GETNEWSTATE.m - Get new state
% 
% This function will return a new state, given a state and action.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param state: certain state.
% @param action: certain option.
% @param newState: new state.

function [ newState ] = getNewState( state, action)

    switch action
        case 1
            newState = state-6;
        case 2
            newState = state+1;
        case 3
            newState = state+6;
        case 4
            newState = state-1;
    end
    

end

