% ACTIVATION.m - Activation function
% 
% This code is the activation function of a neural network. Given a certain
% input and the type of activation function, it will decide an output
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param inout: input to be treated.
% @param type: type of activation function (step or sigmoid).
% @return output: output.

function [ output ] = activation( input, type )
    if strcmp(type,'step')
       if(input>0)
           output=1;
       else
           output=0;
       end
    elseif strcmp(type,'sigmoid')
        output=1/(1+exp(-input));
    elseif strcmp(type,'Dsigmoid') % derivative of the previous sigmoid
        output=activation(input,'sigmoid')*(1-activation(input,'sigmoid'));
    elseif strcmp(type,'tansigmoid')
        output=tanh(input);
    elseif strcmp(type,'linear')
        output=input;
    end
end

