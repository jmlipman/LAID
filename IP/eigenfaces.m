% EIGENFACES.m - Eigenfaces algorithm
% 
% This code is an implementation of the eigenfaces algorithm.
% It uses a training and test set that can be found in github.
% Results a bit unsuccessful since only 16 images are used in the
% training set, but it shows how it works.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%

clear;clc;

ts.name = {'im1.png', 'im2.png', 'im3.png', 'im4.png','im5.png', 'im6.png', 'im7.png' 'im8.png','im9.png', 'im10.png', 'im11.png' 'im12.png', 'im13.png', 'im14.png', 'im15.png' 'im16.png'};
ts.rows = 235;
ts.columns = 235;
ts.length = length(ts.name);

% Read images and vectorize them
for a=1:ts.length
    ts.image{a} = reshape(double(imread(ts.name{a})),1,ts.rows*ts.columns);
end

% Calculate the mean
ts.mean = zeros(1,ts.rows*ts.columns);
for a=1:ts.length
    ts.mean = ts.mean + (1/ts.length)*ts.image{a};
end

% Normalize vectors
for a=1:ts.length
    ts.norm{a} = ts.image{a}-ts.mean;
end


% Generate A matrix
A = zeros(ts.length,ts.rows*ts.columns);
for a=1:ts.length
    A(a,:) = ts.norm{a};
end

% MxM matrix
Cov = A*A';

[eigenvectors,eigenvalues]=eig(Cov);

% Eigenfaces 1. The first is the best because they are (supposed to be
% sorted)
for a=0:ts.length-1
    eigenfaces.image{a+1} = eigenvectors(:,end-a)'*A;
end

% Normalize eigenfaces
%
for a=1:ts.length
    eigenfaces.image{a} = eigenfaces.image{a}./norm(eigenfaces.image{a},2);
end
%

% Eigenfaces 2
%{
Vlarge = A'*eigenvalues;
for a=1:ts.length
    eigenfaces.image{a} = Vlarge(:,a)';
end
%}


% Heuristics to select certain amount of eigenvectors
% 1: Select those until the cumulative sum is around 95%
% eigval = diag(eigenvectors);
% eigval = eigval(end:-1:1);
% eigsum = sum(eigval); 
% csum = 0; 
% for i = 1:size(eigval,1)
%     csum = csum + eigval(i); 
%     tv = csum/eigsum; 
%     if tv > 0.95 
%         k95 = i; 
%         break 
%     end ;
% end;
% 2: Select those eigenvectors whose eigenvalues are 1 or greater
% L_eig_vec = [];
% for i = 1 : size(V,2) 
%     if( D(i,i) > 1 )
%         L_eig_vec = [L_eig_vec V(:,i)];
%     end
% end

% We can adjust manually to use the Nth best eigenvectors
eigenfaces.used = 8;

% Weights
for a=1:ts.length
    for b=1:eigenfaces.used
        ts.w(a,b) = eigenfaces.image{b}*ts.norm{a}';
    end
end


% <<Classification>>

test.name = {'im1.png', 'test1.png', 'test2.png', 'test3.png','test4.png', 'test5.png', 'test6.png','test7.png', 'test8.png', 'test9.png'};
test.rows = 235;
test.columns = 235;
test.length = length(test.name);

% Read images and vectorize them
for a=1:test.length
    test.image{a} = reshape(double(imread(test.name{a})),1,test.rows*test.columns);
end

% Normalize vectors with respect to the training set mean
for a=1:test.length
    test.norm{a} = test.image{a}-ts.mean;
end


% Weights
for a=1:test.length
    for b=1:eigenfaces.used
        test.w(a,b) = eigenfaces.image{b}*test.norm{a}';
    end
end


% Calculate the distances
for a=1:test.length
    for b=1:ts.length
        test.dist(a,b) = sqrt(sum((test.w(a,:)-ts.w(b,:)).^2));
    end
end

for a=1:test.length
    fprintf('%s: %i\n',test.name{a},sum(test.dist(a,:)));
end


