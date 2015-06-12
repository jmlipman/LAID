% SVM.m - Support Vector Machine implementation.
% 
% This code tries to clarify how SVMs work through a clear implementation
% that is explained on a pair of entries on my personal blog. I tried to
% explain as much as I could directly on the code, but when you see a *N*
% it means that I explained it more extensively on the blog, in the 
% Source Code Leyend. As this code is not really easy to follow and
% understand, I left some prints (commented) to help those who want to
% trace how it works.
% The algorithm has been improved:
%       - It focus only on those samples which are important.
%       - There is a maximum iterations allowing it will not iterate
%       always.
%       - It can plot and print the distances from each point to the
%       boundary.
%       - It can generate many pseudorandom samples.
%
% POST URL: http://laid.delanover.com/svm-matlab-code-implementation-smo-sequential-minimal-optimization-and-quadratic-programming-explained
% POST URL: 
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%

% --> What we can improve in the algorithm is the way we choose i and j
% I think I can improve it since it's checking twice the same all the time
% like i=1,j=2 and later on, i=2,j=1. Is this actually the same?

% These samples have only one target out which is the last column.

clear;clc;
%
data = [0 0 3 -1;
        0 3 3 -1;
        3 0 0 1;
        3 3 0 1];
    
%{
data = [-1 -1 -1;
        2 0 1;
        0 2 1;
        3 1 1;];
%}

data =[
    -7    -4    -1;
    -9    -8    -1;
     2     5    -1;
    -3   -10    -1;
     9     7     1;
     3     8     1;
     8    11     1;
     8     9     1;];

% Some pseudorandom samples
%{
sampleNum = 10;
for a=1:sampleNum+1
    data(a,1) = randi([-10 2],1,1);
    data(a,2) = randi([-10 5],1,1);
    data(a,3) = -1;
end

for a=sampleNum+1:sampleNum*2
    data(a,1) = randi([3 10],1,1);
    data(a,2) = randi([7 12],1,1);
    data(a,3) = 1;
end
%}

% This will plot the result
% This only works for 2D trials
plot2DResult = 1;
original_data = data;

C = 1;
tolerance = 0.000001;
alpha = 0;
b = 0;
max_iter = 5;
counter = 0;

% The main loop will iterate until alpha is consistent or until a maximum
% is reached.
while length(find(alpha==0))>0 && counter<max_iter
    
    alpha
    input = data(:,1:end-1);
    target = data(:,end);
    samplesAmount = length(target);
    alpha = zeros(samplesAmount,1);
    
   for i=1:samplesAmount
       
      % Error of sample i
      % Ei = f(xi)-yi (*1*)
      Ei = sum(alpha.*target.*kernel(input,input(i,:),'normal')) - target(i);  
    
      % First if
      % It ensures that when the error is lower than the tolerance, it
      % will not go ahead (*2*)
      if ((Ei*target(i) < -tolerance) && (alpha(i)<C)) || ((Ei*target(i) > tolerance) && (alpha(i) > 0))
          
          % Search over all j
          for j=1:samplesAmount
             
             % If they are both the same, skip it
             if j~=i
                 
                  % Ej = f(xj)-yj, also (*1*)
                  Ej = sum(alpha.*target.*kernel(input,input(j,:),'normal')) - target(j); 
                  
                  alphaiOld = alpha(i);
                  alphajOld = alpha(j);

                  % Compute L and H (explained in the post)
                  if target(i)==target(j)
                      L = max([0 alpha(i)+alpha(j)-C]);
                      H = min([C alpha(i)+alpha(j)]);
                  else
                      L = max([0 alpha(j)-alpha(i)]);
                      H = min([C C+alpha(j)-alpha(i)]);
                  end
                  
                  %fprintf('i: %i, j: %i, L: %i, H: %i, Ai: %i, Aj: %i, eq: %i, Ei: %i, Ej: %i \n',i, j, L, H, alpha(i), alpha(j), target(i)==target(j), Ei, Ej);
                  
                  % We cannot balance alphai and alphaj, so we skip this
                  % combination
                  if H==L
                     %disp('break1');
                     continue; % continue to next j
                  end

                  % Compute eta
                  eta = 2*kernel(input(j,:),input(i,:),'normal')-kernel(input(i,:),input(i,:),'normal')-kernel(input(j,:),input(j,:),'normal');
                  
                  %fprintf('eta: %i\n',eta);
                  % We break if it's not under 0 because then it's not a
                  % maximum.
                  if eta >= 0
                      %disp('break2');
                      continue;
                  end

                  % Compute and clip alphaj to fulfill KKT constraint that
                  % says that 0 <= alphai <= C
                  alpha(j) = alpha(j) - (target(j)*(Ei-Ej))/eta;
                  %fprintf('alphaJ: %i\n',alpha(j));
                  if alpha(j) > H
                      alpha(j) = H;
                  elseif alpha(j) < L
                      alpha(j) = L;
                  end

                  % If alpha(j) barely changed, there is no need to adjust
                  %  alpha(i) since they're basically the same.
                  if abs(alpha(j)-alphajOld) < tolerance
                      %disp('break3');
                      continue;
                  end

                  % AlphaI is related to the increament of alphaJ
                  alpha(i) = alpha(i) + target(i)*target(j)*(alphajOld-alpha(j));

                  % Compute b
                  b1 = b - Ei - target(i)*(alpha(i)-alphaiOld)*kernel(input(i,:),input(i,:),'normal')-target(j)*(alpha(j)-alphajOld)*kernel(input(i,:),input(j,:),'normal');
                  b2 = b - Ej - target(i)*(alpha(i)-alphaiOld)*kernel(input(i,:),input(j,:),'normal')-target(j)*(alpha(j)-alphajOld)*kernel(input(j,:),input(j,:),'normal');

                  %fprintf('b1: %i, b2: %i\n',b1, b2);
                  if (0 < alpha(i)) && (alpha(i) < C)
                      b = b1;
                  elseif (0 < alpha(j)) && (alpha(j) < C)
                      b = b2;
                  else
                     b = (b1+b2)/2;
                  end

             end
             
          end
          
      end % from the first if
   end
   
   counter=counter+1;
   % We will use only those samples who seem to be useful
   data=data((find(alpha~=0)),:);
   
end

% Calculate final W
totalSum = 0;
for i=1:samplesAmount
    totalSum = totalSum + alpha(i)*target(i)*input(i,:);
end

W = totalSum;
b = target(1) - input(1,:)*W';

disp('--------------------------------')
disp('----------- Results: -----------')
disp('--------------------------------')
alpha
W
b


% Plot results
if plot2DResult==1
    
    % This value depends on the samples
    y_bounds = 10;
    y = [-y_bounds:.1:y_bounds];
    x = [];
    for a=1:length(y)
        x = [x (-W(2)*y(a)-b)/W(1)];
    end

    plot(x,y);
    hold on;
    grid on;
    plotPoint(original_data);
    distance = abs(W(1).*original_data(:,1)+W(2).*original_data(:,2)+b )/sqrt(W(1)^2 + W(2)^2);
    disp('Coordiantes, classes and distance to the boundary from each sample:');
    datadistance = [original_data distance];
    for a=1:size(datadistance,1)
        for b=1:size(datadistance,2)-1
            fprintf('\t%i\t',datadistance(a,b));
        end
        fprintf('%f\n',datadistance(a,end));
    end
end

