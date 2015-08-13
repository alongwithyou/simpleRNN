%This demo uses a single layer Neural Network for handwriting number
%classification. It uses back-propagation algorithm to optimize the
% parameters.
%
% You will need MNIST dataset and two functions to load them in matlab.
%
% MNIST dataset can be found here:
% http://yann.lecun.com/exdb/mnist/
% After download MNIST, extract it inside the folder "data".
% You can use the functions provided by Stanford university to load MNIST
% dataset:
% http://ufldl.stanford.edu/wiki/resources/mnistHelper.zip
%
% The optimization function fmincg is provided by Carl Edward Rasmussen.
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



clear ; close all; clc

% config.generative='off';
config.generative='on';

config.l = 'pred';
% config.l = 'class';

% config.net = 'rnn';
config.net = 'ff';

% config.visual='on';
config.visual='off';

config.iter = 50;

config.lr=0.01;

datadim = 2;
neron = 3;
if strcmp(config.l,'pred')
    if strcmp(config.net,'ff')
        %  Parameters
        input_layer_size  = datadim;  
        first_layer_size = neron;
        second_layer_size = neron;
    elseif strcmp(config.net,'rnn')
        input_layer_size  = datadim; 
        first_layer_size = neron;
        second_layer_size = neron;
    end
    num_labels = datadim;          
    lambda = 1;
    
   
    
    N = 1000;
    %fs1 = 25;
    %fs2 = 50;
    %fs3 = 75;
    %nr = 0.1;
    
    [x1,t1] = generate_ones(N);
    %     [x2,t2] = generate_ones(N);
    
    %   [x1, t] = generate_sine(N,nr,fs1,fs2,fs3);
    %   [x2, t] = generate_sine(N,nr,fs1,fs2,fs3);
    
    %     [x1,t] = generate_sine(N,nr,fs1,fs2);
    %     [x2,t] = generate_sine(N,nr,fs1,fs2);
    
    %     [x1, t] = generate_sine(N,nr,fs1);
    %     [x2, t] = generate_sine(N,nr,fs1);
    
    
    X = [x1 x1];
    y = [t1 t1];
    
elseif strcmp(config.l,'class')
    if strcmp(config.net,'ff')
        %  Parameters
        input_layer_size  = 784;  % 28x28 Input Images of Digits
        first_layer_size = 100;
        second_layer_size = 50;% 25 Hidden units
    elseif strcmp(config.net,'rnn')
        input_layer_size  = 784;  % 28x28 Input Images of Digits
        first_layer_size = 50;
        second_layer_size = 50;% 25 Hidden units
    end
    num_labels = 10;          % 10 Labels, from 1 to 10
    lambda = 1;               % Regularization parameter
    
    % Load training data
    %     [Xtrain, ytrain, Xtest, ytest, Xcross, ycross] = ReadMNIST();
    [Xtrain, ytrain] = ReadMNIST4RNN();
    X=Xtrain;
    y=ytrain;
    %     X = loadMNISTImages('train-images-idx3-ubyte')';
    %     y = loadMNISTLabels('train-labels-idx1-ubyte');
    % Make labels from 1 to 10
    %     y = y+1;
    
    
    
end




m = size(X, 1);

disp('Initialize the parameters ...')

init_theta_first = initW(input_layer_size ,first_layer_size);

init_theta_second = initW(first_layer_size,second_layer_size );
init_theta_output = initW( second_layer_size,num_labels);
% Put the parameters together
initi_params = [init_theta_first(:) ; init_theta_second(:); init_theta_output(:)];

disp('Train the Neural Network (It might take a while) ...')


% Set the maximum iterations
options = optimset('MaxIter', config.iter);



if strcmp(config.net,'ff')
    costFunction = @(p) CostFunction(p, input_layer_size, first_layer_size, second_layer_size, num_labels, X, y, lambda,config);
elseif strcmp(config.net,'rnn')
    costFunction = @(p) CostFunction_RNN(p, input_layer_size, first_layer_size, second_layer_size, num_labels, X, y, lambda,config);
end

% Minimize the cost using fmincg function
[nn_params, cost] = fmincg(costFunction, initi_params, options);

Theta_first = reshape(nn_params(1:first_layer_size * (input_layer_size + 1)), first_layer_size, (input_layer_size + 1));

Theta_second = reshape(nn_params((1 + (first_layer_size * (input_layer_size + 1))):(( (first_layer_size * (input_layer_size + 1))) + (second_layer_size* (first_layer_size+1)))), second_layer_size, (first_layer_size + 1));

Theta_output = reshape(nn_params((( (first_layer_size * (input_layer_size + 1))) + (second_layer_size* (first_layer_size+1)) +1):end), num_labels, (second_layer_size + 1));

config.generative='off';
% config.generative='on';



if strcmp(config.l,'class')
    if strcmp(config.net,'ff')
        pred = PredictLabels(Theta_first, Theta_second, Theta_output, X);
    elseif strcmp(config.net,'rnn')
        pred = PredictLabels_RNN(Theta_first, Theta_second, Theta_output, X,config);
    end
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
elseif strcmp(config.l,'pred')
    if strcmp(config.net,'ff')
        pred = PredictLabels(Theta_first, Theta_second, Theta_output, X);
    elseif strcmp(config.net,'rnn')
        pred = PredictLabels_RNN(Theta_first, Theta_second, Theta_output, X,config);
    end
    pred=pred(:,1);
    y=y(:,1);
    thresh=mean([min(pred) max(pred)]);
    pred = pred > thresh;
    [prec,rec,F1,acc]=calc_measures(pred,y);
    fprintf('\nTraining Set Precision: %f\n',prec * 100);
    fprintf('\nTraining Set Recall: %f\n',rec * 100);
    fprintf('\nTraining Set F1: %f\n',F1 * 100);
    fprintf('\nTraining Set Accuracy: %f\n',acc * 100);
    pause(3);
    % TEST DATA 100 times
    for kk=1:100
    clc
        [x1,t1] = generate_ones(N);
        X = [x1 x1];
        y = [t1 t1];
        
        if strcmp(config.net,'ff')
            pred = PredictLabels(Theta_first, Theta_second, Theta_output, X);
        elseif strcmp(config.net,'rnn')
            pred = PredictLabels_RNN(Theta_first, Theta_second, Theta_output, X,config);
        end
        pred=pred(:,1);
        y=y(:,1);
        thresh=mean([min(pred) max(pred)]);
        pred = pred>thresh;
        [prec,rec,F1,acc]=calc_measures(pred,y);
        fprintf('\nTesting Set Precision: %f\n',prec * 100);
        fprintf('\nTesting Set Recall: %f\n',rec * 100);
        fprintf('\nTesting Set F1: %f\n',F1 * 100);
        fprintf('\nTesting Set Accuracy: %f\n',acc * 100);
        pause(3);
    end
    
end