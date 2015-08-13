function g = sigmoid(z)
%   Sigmoid function
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
% g = 1.0 ./ (1.0 + exp(-z));
g = 1.0 ./ (1.0 + exp(-z));
% G = 1.0 ./ (1.0 + exp(-Z));
% g = max(1.0 ./ (1.0 + exp(-z)),0);

end
