% UPDATEVFIELD.m - Writing V values.
% 
% This function will simply write in the V(s) field all V(s) values.
%
% You can do with this code whatever you want. The main purpose is help
% people learning about this. Also, there is no warranty of any kind.
%
% Juan Miguel Valverde Martinez
% http://laid.delanover.com
%
% @param V: matrix with V values that will be written in the right maze.


function updateVfield( V )

    for a=0:5
       for b=0:5
            
           a_h = annotation('textbox','string', V(6-b,a+1), 'tag', 'tmptext', 'edgecolor', 'white');
           set(a_h, 'units', 'pixels', 'position', [331+50*a 45+50*b 10 10]);
           
       end
    end

end

