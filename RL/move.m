% MOVE.m - Eager goalkeeper
% 
% This function will move the player from one position to another.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param player_h: player representation (the rectangle).
% @param positionPlayer: player's positon.
% @param distance: distance to the next position.
% @param playerHeight: height of the rectangle representing the player.
% @param totalPositons: total amount of positions.
% @return positionPlayer: new player's position.


function [positionPlayer] = move( player_h, positionPlayer, distance, playerHeight, totalPositions )

    if distance~=0
        positionPlayer = positionPlayer + distance;
        if positionPlayer>totalPositions-1
            positionPlayer=totalPositions-1;
        elseif positionPlayer<0
            positionPlayer=0
        end
        set(player_h, 'position', [.05,.3+playerHeight*(positionPlayer),.02,playerHeight]); 
    end
    
end

