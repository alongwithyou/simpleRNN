function RW = initW(dim1, dim2)
% Generates the weight matrix.
% 
%      Copyright (C) Hamid 2015
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation;
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    EPS = 0.12;
    EPS = 0.22;
%     W = rand(dim1, 1 + dim2) * 2 * EPS - EPS;
    RW = rand(dim2, 1 + dim1) * 2 * EPS - EPS;
end
