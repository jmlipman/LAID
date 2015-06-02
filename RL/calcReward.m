% CALCREWARD.m - Calc reward
% 
% This function returns the reward given a state and action.
% As we do not know when it will find its goal, it will always have a
% reward of 0 unless it finds the goal.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param state: certain state.
% @param action: certain action.
% @return reward: corresponding reward.

function [ reward ] = calcReward( state, action )

    if ((state==30 && action==3) || (state==35 && action==2))
        reward = 2;
    else
        reward = 0;
    end

end

