% SVM.m - Support Vector Machine implementation.
% 
% This code tries to clarify how SVMs work through a clear implementation
% that is explained on a pair of entries on my personal blog. I tried to
% explain as much as I could directly on the code, but when you see a *N*
% it means that I explained it more extensively on the blog, in the 
% Source Code Leyend. As this code is not really easy to follow and
% understand, I left some prints (commented) to help those who want to
% trace how it works.
% POST URL: http://laid.delanover.com/svm-matlab-code-implementation-smo-sequential-minimal-optimization-and-quadratic-programming-explained
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


input = data(:,1:end-1);
target = data(:,end);
samplesAmount = length(target);

C = 1;
tolerance = 0.000001;
alpha = zeros(samplesAmount,1);
b = 0;
max_passes = 1;
passes = 0;

while passes<max_passes
   num_changed_alphas = 0;
   
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
                      %disp('-------------------1');
                  elseif alpha(j) < L
                      alpha(j) = L;
                      %disp('-------------------2');
                  end

                  % If alpha(j) barely changed, there is no need to adjust
                  %  alpha(i) since they're basically the same.
                  if abs(alpha(j)-alphajOld) < tolerance
                      %disp('break3');
                      continue;
                  end

                  % AlphaI is related to the increament of alphaJ
                  alpha(i) = alpha(i) + target(i)*target(j)*(alphajOld-alpha(j));


                  %fprintf('alphaI: %i\n',alpha(i));
                  b1 = b - Ei - target(i)*(alpha(i)-alphaiOld)*kernel(input(i,:),input(i,:),'normal')-target(j)*(alpha(j)-alphajOld)*kernel(input(i,:),input(j,:),'normal');
                  b2 = b - Ej - target(i)*(alpha(i)-alphaiOld)*kernel(input(i,:),input(j,:),'normal')-target(j)*(alpha(j)-alphajOld)*kernel(input(j,:),input(j,:),'normal');
                  % Compute b

                  %fprintf('b1: %i, b2: %i\n',b1, b2);
                  if (0 < alpha(i)) && (alpha(i) < C)
                      %disp('in1');
                      %disp(alpha(i));
                      b = b1;
                  elseif (0 < alpha(j)) && (alpha(j) < C)
                      %disp('in2');
                      b = b2;
                  else
                      %disp('in3');
                     b = (b1+b2)/2;
                  end

                   
                  num_changed_alphas = num_changed_alphas+1;
                 
             end
             
          end
          
      end % from the first if
   end
   
   if num_changed_alphas==0
       passes = passes+1;
   else
       passes = 0;
   end
   
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


