% NN233.m - 2x3x3 NN
%
% This code implements a 2x3x3 MLP in order to clarify how it works.
% Depending on the learning rate and iterations, we can clearly see the
% error rate decreasing, so it is able to classify the training set
% perfectly, but it will probably overfit it.
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

% This is just a MLP with 2 inputs, one output and 3 hidden nodes in its
% middle hidden layer
clear;clc;

% Samples per class. The more, the more likely to not converge
eachGroup=15;
% Amount of output neurons
totalClasses=3;

bias=-1;
% 2 input values to represent 3 different classes
data = 0.5 + (4-0.5).*rand(eachGroup,2);
data = [data zeros(eachGroup,2) ones(eachGroup,1)];
 
data2 = 5 + (10-5).*rand(eachGroup,2);
data2 = [data2 zeros(eachGroup,1) ones(eachGroup,1) zeros(eachGroup,1)];
 
data31 = 6 + (9-6).*rand(eachGroup,1);
data32 = 0 + (5-0).*rand(eachGroup,1);
data3 = [data31 data32 ones(eachGroup,1) zeros(eachGroup,1) zeros(eachGroup,1)];
 
data = [data;data2;data3];
 
% Input = bias + the rest without last row (targets)
input = [bias*ones(size(data,1),1),data(:,1:end-totalClasses)];
% Last N columns are now target
target = data(:,end-(totalClasses-1):end);

% Feedforward part
% Random weights between the input and hidden layer
% First row corresponds to the input to the first hidden node
weights1 = rand(3,3);

% Random weights between the hidden layer and the output
weights2 = rand(3,4);
% Learning rate
eta = -0.1;
% Times the algorithm will iterate
% An alternative is to iterate until the error is below a number or until
% the difference between the present and previous error is epsilon.
learningTimes=100;



for t=1:learningTimes

    % This loop could be avoided if all inputs were given to the NN at once
    % But to make it easier to understand, I chose this way.
    for tt=1:length(input)
    
        % Input elements from each individual sample
        indSample = input(tt,:);
        % Desired output
        y = target(tt,:);
        
        z2 = indSample*weights1';

        % Get the activation in each element of the array
        a2=arrayfun(@(x) activation(x,'sigmoid'),z2);
        % Add the bias to the current activation
        a2=[bias a2];

        z3=a2*weights2';

        y_hat=arrayfun(@(x) activation(x,'sigmoid'), z3);

        J = 0.5*sum((y-y_hat).^2);

        % Now we calculate the derivative of the squared error (J) respect to
        % weights2:

        d3 = -1*(y-y_hat).*arrayfun(@(x) activation(x,'Dsigmoid'), z3);
        dJ2 = a2'*d3;

        % Let's remove the bias from the hidden layer as well
        dJ1_tmp = d3*weights2(:,2:end);
        dJ1 = indSample'*(dJ1_tmp.*arrayfun(@(x) activation(x,'Dsigmoid'), z2));

        weights1 = weights1 + eta*dJ1;
        weights2 = weights2 + (eta*dJ2)';
        
    end
    
    fprintf('Error: %f\n', J);
    FJ(t) = J;
    
end
