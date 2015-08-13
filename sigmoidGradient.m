function g = sigmoidGradient(z)
g = sigmoid(z) .* (1 - sigmoid(z));
% G = (Sigmoid(Z) .* (1-Sigmoid(Z)));
end