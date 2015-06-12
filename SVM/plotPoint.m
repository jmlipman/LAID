% PLOTPOINT.m - Plot point function
% 
% This function assists SVM in order to plot points given the whole data
% set. Depending on the class, it will be plotet in a different manner,
% typically with a blue/red cross and circle.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param input: input samples which have coordinates and the class they
% belong to.

function [] = plotPoint( input )

    targets = input(:,end);
    total = size(input,1);
    
    for a=1:total
        if targets(a)==1
            plot(input(a,1),input(a,2),'r+')
            plot(input(a,1),input(a,2),'ro')
        else
            plot(input(a,1),input(a,2),'b+')
            plot(input(a,1),input(a,2),'bo')
        end
    end

end

