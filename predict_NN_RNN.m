function [p] = predict_NN_RNN(theta1, theta2,theta3, X,X_TM1,config)
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
num_labels = size(theta3, 1);

% initial the predictions
p = zeros(size(X, 1), 1);

% % first layer
% Add bias term
d1 =X;
d2=theta1*[ones(m, 1) d1]';
d2=sigmoid(d2);

d1_tm1 = X_TM1;
d2_tm1 = theta1 * [ones(m, 1) d1_tm1]';
d2_tm1 = sigmoid(d2_tm1);

% d1_tm2 = X_TM2;
% d2_tm2 = theta1 * [ones(m, 1) d1_tm2]';
% d2_tm2 = sigmoid(d2_tm2);
% 
% d1_tm3 = X_TM3;
% d2_tm3 = theta1 * [ones(m, 1) d1_tm3]';
% d2_tm3 = sigmoid(d2_tm3);

% % % % % % % % % % % % % % % % % % % % % 
d3_tm1 = theta2 * [ones(m, 1), d2_tm1']';
d3_tm1 = sigmoid(d3_tm1);

% d3_tm2 = theta2 * [ones(m, 1), d2_tm2']';
% d3_tm2 = sigmoid(d3_tm2);
% 
% d3_tm3 = theta2 * [ones(m, 1), d2_tm3']';
% d3_tm3 = sigmoid(d3_tm3);

% % % % % % % % % % % % % % % % % % % % % % 
% d4_tm2 = theta3 * [ones(m, 1), d3_tm2']';
% d4_tm2 = sigmoid(d4_tm2);

% d4_tm3 = theta3 * [ones(m, 1), d3_tm3']';
% d4_tm3 = sigmoid(d4_tm3);
% % % % % % % % % % % % % % % % % % % % % % % 
% d5_tm3 = theta4 * [ones(m, 1), d4_tm3']';
% d5_tm3 = sigmoid(d5_tm3);


% Output layer
% Add bias term
d6=theta3*[ones(1,m) ;d2+d3_tm1];
d6=sigmoid(d6);

if strcmp(config.l,'class')
% pick the maximum
[~,ix]=max(d6);
p = ix;
elseif strcmp(config.l,'pred')
    p=d6;
end
    
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