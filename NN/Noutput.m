% Noutput.m - 2xN NN which may not converge
% 
% This code implements a perceptron with N output neurons. The final result
% is not the best one.
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
eachGroup=15;
% Amount of output neurons
totalClasses=2;
 
data = 0.5 + (4-0.5).*rand(eachGroup,2);
data = [data zeros(eachGroup,2) ones(eachGroup,1)];
 
data2 = 5 + (10-5).*rand(eachGroup,2);
data2 = [data2 zeros(eachGroup,1) ones(eachGroup,1) zeros(eachGroup,1)];
 
data31 = 6 + (9-6).*rand(eachGroup,1);
data32 = 0 + (5-0).*rand(eachGroup,1);
data3 = [data31 data32 ones(eachGroup,1) zeros(eachGroup,1) zeros(eachGroup,1)];
 
data = [data;data2;data3];
 
% Input = bias + the rest without last row (targets)
input = [-1*ones(size(data,1),1),data(:,1:end-totalClasses)]
% Last N columns are now target
target = data(:,end-(totalClasses-1):end)
 
%Learning rate
eta = 0.5;
% Amount of features. -N last rows
inputNeurons = size(data,2)-totalClasses;
% Amount of inputs for training (4 cases)
inputCases = size(data,1);
 
 
disp('First weights');
weights = rand(inputNeurons+1,totalClasses)';
disp(weights);
 
% Plot elements
plot(input(1:eachGroup,2),input(1:eachGroup,3),'mo')
hold on;
plot(input(eachGroup+1:eachGroup*2,2),input(eachGroup+1:eachGroup*2,3),'ro')
plot(input(eachGroup*2+1:eachGroup*3,2),input(eachGroup*2+1:eachGroup*3,3),'bo')
 
% Training 10 times
repeat=1;
counter=1;
while repeat==1
    repeat=0;
    fprintf('Counter: %i.\n',counter) 
        
    for j=1:inputCases
        
        for k=1:totalClasses
            y=activation(input(j,:)*weights(k,:)','step');
            
            if target(j,k)~=y
            %if abs(target(j)-y1)>0.5 % We have to use this with sigmoid since
            %it's regression
               fprintf('Target: %i, y: %i, j: %i\n',target(j,k),y,j);
               weights(k,:) = weights(k,:) + eta*(target(j,k)-y)*input(j,:);
               repeat=1;
            end
 
        end
    end
    counter=counter+1;
       
end
disp('Final weights');
disp(weights);
 
 
% Ploting the mesh
for i=-0.5:0.3:10.5
   for j=-0.5:0.3:10.5
       % Change this if using +1 as bias
       for k=1:totalClasses
            res(k)=activation(-weights(k,1)+weights(k,2)*i+weights(k,3)*j,'step');
       end
       index=find(res); %it finds the index of non-zero values
       
       %if we change totalClasses, we have to change this to make it work
        if index==1
            plot(i,j,'ro');
        elseif index==2
            plot(i,j,'bo');
        else
            plot(i,j,'mo');  
        end
   end 
end
 
x = -3:13;
% Decision boundary
% Change this if using +1 as bias
for i=1:totalClasses
    y = (-weights(i,2)*x+weights(i,1))/weights(i,3);
    plot(x,y);
end
 
plot(input(:,2),input(:,3),'ko')
hold off;
