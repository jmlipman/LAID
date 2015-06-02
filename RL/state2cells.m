% STATE2CELLS.m - Convert states into cell coordinates.
% 
% This function converts from a certain state to its corresponding row and
% column equivalence.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param state: state to be converted.
% @return row: equivalent row coordinate in the maze.
% @return column: equivalent column coordinate in the maze.


function [ row, column ] = state2cells( state )

    row = ceil(state/6);
    column = mod(state-1,6)+1;

end

