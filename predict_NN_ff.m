function [p,d4] = predict_NN_ff(theta1, theta2, X)
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
m = size(X, 1);
num_labels = size(theta2, 1);

% initial the predictions
p = zeros(size(X, 1), 1);

% % first layer
% Add bias term
d1=theta1*[ones(m, 1) X]';
d2=sigmoid(d1);

% Output layer
% Add bias term
d3=theta2*[ones(1,m) ;d2];
d4=sigmoid(d3);

% pick the maximum
[~,ix]=max(d4);
p = ix;
    
% loop version. commented  
% for i=1:size(X,1)
%     data = X(i,:);
%     
% % Add bias term
%     d1=theta1*[1 data]';
%     d2=sigmoid(d1);
%     
% % Add bias term
%     d3=Theta2*[1 ;d2];
%     d4=sigmoid(d3);
%     
% % pick the maximum
%     [~,ix]=max(d4);
%     p(i) = ix;
% end


end