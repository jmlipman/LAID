% AND_OR_XOR.m - 2x1 NN with basic operations
% 
% This code implements a very simple perceptron with 2 input neurons (+
% bias) and an output neuron. Basic operations as And, Or and Xor will be
% used as examples to illustrate the functioning of a perceptron.
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

clear;clc;
 
% AND
data = [0 0 0; 0 1 0; 1 0 0; 1 1 1];
% OR
%data = [0 0 0; 0 1 1; 1 0 1; 1 1 1];
% XOR (it will never converge)
%data = [0 0 0; 0 1 1; 1 0 1; 1 1 0];
 
 
% Input = bias + the rest without last row (targets)
input = [-1*ones(size(data,1),1),data(:,1:end-1)];
target = data(:,end);
 
%Learning rate
eta = 0.5;
% Amount of features
inputNeurons = size(data,2)-1;
% Amount of inputs for training (4 cases)
inputCases = size(data,1);
 
disp('First weights');
weights = rand(inputNeurons+1,1)';
disp(weights);
 
% Plot elements
plot(input(:,2),input(:,3),'ro')
hold on;
 
% Training until it has no errors. The counter can be used to establish a
% maximum and set "repeat" flag as 0.
repeat=1;
counter=1;
while repeat==1
    repeat=0;
    %fprintf('Counter: %i.\n',counter) 
    for j=1:inputCases
        y=activation(input(j,:)*weights','step');
        if target(j)~=y
           %fprintf('Target: %i, y: %i, j: %i\n',target(j),y,j);
           weights = weights + eta*(target(j)-y)*input(j,:);
           repeat=1;
        end
        
    end
    counter=counter+1;
    
end
disp('Final weights');
disp(weights);
 
% Ploting the mesh
for i=-0.5:0.05:1.5
   for j=-0.5:0.05:1.5
  % Change this if using +1 as bias
        res=activation(-weights(1)+weights(2)*i+weights(3)*j,'step');
        if res==1
            plot(i,j,'ro');
        else
            plot(i,j,'bx');
        end
   end 
end
 
x = -3:3;
% Decision boundary
% Change this if using +1 as bias
y = (-weights(2)*x+weights(1))/weights(3);
plot(x,y);
hold off;
