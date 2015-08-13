function [Cost Grad] = CostFunction_RNN(Parameters, input_layer_size, hidden_layer_size1, hidden_layer_size2, num_labels, X, y, lambda,config)

% get theta values
ThetaI = reshape(Parameters(1:hidden_layer_size1 * (input_layer_size + 1)), hidden_layer_size1, (input_layer_size + 1));

ThetaR = reshape(Parameters((1 + (hidden_layer_size1 * (input_layer_size + 1))):(( (hidden_layer_size1 * (input_layer_size + 1))) + (hidden_layer_size2* (hidden_layer_size1+1)))), hidden_layer_size2, (hidden_layer_size1 + 1));

ThetaO = reshape(Parameters((( (hidden_layer_size1 * (input_layer_size + 1))) + (hidden_layer_size2* (hidden_layer_size1+1)) +1):end), num_labels, (hidden_layer_size2 + 1));
m = size(X, 1);

Cost = 0;

ThetaI_grad = zeros(size(ThetaI));
ThetaR_grad = zeros(size(ThetaR));
ThetaO_grad = zeros(size(ThetaO));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nR1 = 1;
nC1 = size(X,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a1 = [ones(m, 1) X];
z2 = ThetaI * a1';
a2 = sigmoid(z2);
a2 = a2';
a2 = [ones(nC1,nR1) a2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     	z3 = ThetaR * a2';
%     	a3 = sigmoid(z3);
%     	a3 = a3';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a1r = [ones(m, 1) [zeros(1,size(X,2)); X(2:end,:)]];
z2r = ThetaI * a1r';
a2r = sigmoid(z2r);
a2r = a2r';

a2r = [ones(nC1,nR1) a2r];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

z3r = ThetaR * a2r';
a3r = sigmoid(z3r);
a3r = a3r';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nR2 = 1;
nC2 = size(a2,1);
a3r = [ones(nC2,nR2) a3r];
z4 = ThetaO * (a3r'+a2');
a4 = sigmoid(z4 );
a4 = a4';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(config.l,'class')
    for i = 1:size(y,1)
        g = find(y == y(i));
        Y(g,y(i)) = 1;
    end
elseif strcmp(config.l,'pred')
    Y=y;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%filling up y vector.
% 1 is represented as 1x10 vector as in 100000...
% 2 is represented as 1x10 vector as in 010000...

AA = ((-Y .* log(a4)) - ((1-Y) .* log(1 - a4)))';
BB = sum(AA);

Cost = sum(BB);
Cost = Cost/m;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% regularizing to reduce overfitting

RP = (lambda/(2*m));

LI  = ThetaI(:,2:end)';
LR = ThetaR(:,2:end)';
LO = ThetaO(:,2:end)';

regularized_parameter = RP * ( sum((sum(LI .* LI))) + sum((sum(LR .* LR)))  + sum((sum(LO .* LO))) );

Cost = Cost + regularized_parameter;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % sigmoidGradient back prop

for t = 2:m
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    a1 = X(t, :)';
    a1 = [1; a1];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    a1 = [1; X(t,:)'];
    z2 = ThetaI * a1;
    a2 = sigmoid(z2);
    a2 = [1 ;a2];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    a1r = [1; X(t-1,:)'];
    z2r = ThetaI * a1r;
    a2r = sigmoid(z2r);
    a2r = [1 ;a2r];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    z3r = (ThetaR * a2r);
    a3r = sigmoid(z3r);
    a3r = [1; a3r];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    z4 = (ThetaO * (a3r+a2));
    a4 = sigmoid(z4);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    HThetaX = a4;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Y_t = Y(t,:)';
    delta_O = (HThetaX - Y_t);
    % difference in output = delta_O
    
    delta_R = ((ThetaO(:, 2:end)' * delta_O) .* (sigmoidGradient(z3r)+sigmoidGradient(z2)));
    
    delta_I = ((ThetaR(:, 2:end)' * delta_R) .* (sigmoidGradient(z2r)));
    
    ThetaO_grad = ThetaO_grad + (delta_O * (a3r' +  a2'));
    ThetaR_grad = ThetaR_grad + (delta_R * a2r');
    ThetaI_grad = ThetaI_grad + (delta_I * a1');
end
ThetaI_grad = (ThetaI_grad/(m));
ThetaR_grad = (ThetaR_grad/(m));
ThetaO_grad = (ThetaO_grad/(m));

ThetaI_grad(:,2:end) = (ThetaI_grad(:,2:end) + (config.lr * (lambda/(m)) * ThetaI(:,2:end)));
ThetaR_grad(:,2:end) = (ThetaR_grad(:,2:end) + (config.lr * (lambda/(m)) * ThetaR(:,2:end)));
ThetaO_grad(:,2:end) = (ThetaO_grad(:,2:end) + (config.lr * (lambda/(m)) * ThetaO(:,2:end)));
if strcmp(config.visual,'on')
    if strcmp(config.l,'pred')
        pred=[];
        
        %         pred(t,:)=predict_NN_RNN(ThetaI + ThetaI_grad, ThetaR+ThetaR_grad,ThetaO+ThetaO_grad, X(1,:),zeros(size(X(1,:))),config);
        %         for t = 2:m
        %             pred(t,:)=predict_NN_RNN(ThetaI + ThetaI_grad, ThetaR+ThetaR_grad,ThetaO+ThetaO_grad, X(t,:),X(t-1,:),config);
        %         end
        
        
        pred(t,:)=predict_NN_RNN( ThetaI+ThetaI_grad, ThetaR+ThetaR_grad,ThetaO+ThetaO_grad, X(1,:),zeros(size(X(1,:))),config);
%         pred(t,:)=  PredictLabels_RNN(Theta_first, Theta_second, Theta_output, X,config);
        for t = 2:m
            pred(t,:)=predict_NN_RNN( ThetaI+ThetaI_grad, ThetaR+ThetaR_grad,ThetaO+ThetaO_grad, X(t,:),X(t-1,:),config);
        end
        
        
        plot(pred,'r');
        hold on
        plot(X)
        hold off;
        legend({'pred','real'});
        drawnow;
    elseif strcmp(config.l,'class')
        subplot(3,1,1)
        imagesc(ThetaI);
        title('hidden 1');
        
        subplot(3,1,2)
        imagesc(ThetaR);
        title('recurrent');
        
        subplot(3,1,3)
        imagesc(ThetaO);
        title('output');
        drawnow;
    end
end

Grad = [ThetaI_grad(:) ; ThetaR_grad(:); ThetaO_grad(:)];

end