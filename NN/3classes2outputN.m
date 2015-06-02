% 3CLASSES2OUTPUTN.m - 2x2 NN which may not converge
% 
% This code implements a perceptron with 2 output neurons and three
% different classes to classify, despite that it only has 2 output neurons.
% This code is an example of a problem whose data may not converge because
% it can be non-linear separable, but sometimes it works (pure luck).
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
 
% Samples per class. The more, the more likely to not converge
eachGroup=6;
 
data = 0.5 + (4-0.5).*rand(eachGroup,2);
data = [data zeros(eachGroup,2)];
 
data2 = 5 + (10-5).*rand(eachGroup,2);
data2 = [data2 zeros(eachGroup,1) ones(eachGroup,1)];
 
data31 = 6 + (9-6).*rand(eachGroup,1);
data32 = 0 + (5-0).*rand(eachGroup,1);
data3 = [data31 data32 ones(eachGroup,1) zeros(eachGroup,1)];
 
data = [data;data2;data3];
 
 
% Input = bias + the rest without last row (targets)
input = [-1*ones(size(data,1),1),data(:,1:end-2)]
% Last two columns are now target
target = data(:,end-1:end);
 
%Learning rate
eta = 0.5;
% Amount of features. -2 last rows
inputNeurons = size(data,2)-2;
% Amount of inputs for training (4 cases)
inputCases = size(data,1);
 
 
disp('First weights');
weights1 = rand(inputNeurons+1,1)';
weights2 = rand(inputNeurons+1,1)';
disp(weights1);
disp(weights2);
 
% Plot elements
plot(input(:,2),input(:,3),'go')
hold on;
 
% Training 10 times
repeat=1;
counter=1;
while repeat==1
    repeat=0;
    %fprintf('Counter: %i.\n',counter) 
    for j=1:inputCases
        % 2 different output neurons
        y1=activation(input(j,:)*weights1','step');
        y2=activation(input(j,:)*weights2','step');
        
        if target(j,1)~=y1
           %fprintf('Target: %i, y: %i, j: %i\n',target(j,1),y1,j);
           weights1 = weights1 + eta*(target(j,1)-y1)*input(j,:);
           repeat=1;
        end
        
        if target(j,2)~=y2
           %fprintf('Target: %i, y: %i, j: %i\n',target(j,2),y2,j);
           weights2 = weights2 + eta*(target(j,2)-y2)*input(j,:);
           repeat=1;
        end
    end
    counter=counter+1;
    
       
end
disp('Final weights');
disp(weights1);
disp(weights2);
 
 
% Ploting the mesh
for i=-0.5:0.3:10.5
   for j=-0.5:0.3:10.5
       % Change this if using +1 as bias
        res1=activation(-weights1(1)+weights1(2)*i+weights1(3)*j,'step');
        res2=activation(-weights2(1)+weights2(2)*i+weights2(3)*j,'step');
        
        if res1==0 && res2==0
        %if res>0.5 % regresion
            plot(i,j,'ro');
        elseif res1==0 && res2==1
            plot(i,j,'bo');
        else
            plot(i,j,'mo');
        end
   end 
end
 
x = -3:13;
% Decision boundary
% Change this if using +1 as bias
y = (-weights1(2)*x+weights1(1))/weights1(3);
plot(x,y);
y = (-weights2(2)*x+weights2(1))/weights2(3);
plot(x,y);
plot(input(:,2),input(:,3),'ko')
hold off;
