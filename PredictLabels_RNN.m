function [h3] = PredictLabels_RNN(Theta1, Theta2, Theta3, XO,config)
m=size(XO,1);
dval=[];
prob=[];

i=1;
X=XO(i,:);
if strcmp(config.generative,'off')
    Xt=zeros(size(XO(i,:)));
elseif strcmp(config.generative,'on')
    Xt=XO(i,:);
end
h1 = sigmoid([ones(size(X, 1), 1) X] * Theta1');

h1t = sigmoid([ones(size(X, 1), 1) Xt] * Theta1');

h2t = sigmoid([ones(size(Xt, 1), 1) h1t] * Theta2');

h3(i,:) = sigmoid( ([ones(size(X, 1), 1) h2t] * Theta3') + ([ones(size(Xt, 1), 1) h1] * Theta3'));
% [dval(i,:), prob(i,:)] = max(h3, [], 2);

for i=2:m
    X=XO(i,:);
    if strcmp(config.generative,'off')
        Xt = XO(i-1,:);
    elseif strcmp(config.generative,'on')
        Xt = h3(i-1,:);
    end
    h1 = sigmoid([ones(size(X, 1), 1) X] * Theta1');
    
    h1t = sigmoid([ones(size(X, 1), 1) Xt] * Theta1');
    
    h2t = sigmoid([ones(size(Xt, 1), 1) h1t] * Theta2');
    
    h3(i,:) = sigmoid( ([ones(size(X, 1), 1) h2t] * Theta3') + ([ones(size(Xt, 1), 1) h1] * Theta3'));
    %     [dval(i,:), prob(i,:)] = max(h3, [], 2);
end

end