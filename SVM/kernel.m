% SVM.m - Support Vector Machine implementation.
% 
% This function performs the kernel function. If we have two matrices: A
% matrix is 4x3 and B is 1x3, this function will iterate over matrix A to
% convolve each row with B matrix. Hence, the result will be a 4x1 matrix.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param wholeInputs: matrix A to be iterated.
% @param input: matrix which will iterate over matrix A to do convolution.
% @param type: there might be more types of kernel function in the future.
% @return: result matrix of the convolution.


function [ output ] = kernel(wholeInputs, input, type)

    if strcmp(type,'normal')
       times = size(wholeInputs,1);
       for a=1:times
          output(a,:) = wholeInputs(a,:)*input';
       end
       
    end

end

