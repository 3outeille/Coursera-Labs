function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%-------------------------------------------------------------
%Feedforward and Cost Function.
y_vec = [1:num_labels] == y; %(5000 x 10)

% X = (5000 x 401)
% y = (5000 x 1)
% y_vec = (5000, 10)
% Theta1 = (25 x 401)
% Theta2 = (10 x 26)
% m = 5000

%From input layer to hidden layer.
X = [ones(m, 1) X];
a2 = [ones(m, 1) sigmoid(X*Theta1')]; %add bias a^2_0 -> (5000 x 26).

%From hidden layer to output layer.
h = sigmoid(a2*Theta2'); %(5000 x 10)

cost = -y_vec.*log(h) - (1 -y_vec) .* log(1 - h);

J = (1/m)*sum(sum(cost));
% -------------------------------------------------------------
%Regularized cost function.

%Remove bias (first column).
%(25 x 400).
Theta1_noBias = Theta1(:, 2:end);
%(10 x 25).
Theta2_noBias = Theta2(:, 2:end);

regularization_term = (lambda/(2*m))*(sum(sum(Theta1_noBias.^2)) + sum(sum(Theta2_noBias.^2)));

J += regularization_term;
% =========================================================================
%Backpropagation.
for t = 1:m
%STEP 1: Set input layer to the t-th training exanmple x(t).
  %(401, 1).
  a1 = [X(t, :)']; 
  %Feedforward.
  %(25 x 1) = (25 x 401) * (401 x 1).
  z2 = Theta1 * a1;
  %(26 x 1).
  a2 = [1; sigmoid(z2)]; 
  %(10 x 1) = (10 x 26) * (26 x 1)
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
 
  %(10 x 1)
  y_vec = ([1:num_labels] == y(t))'; 
  
%STEP 2: Compute error in last layer.
  %(10 x 1) = (10 x 1) - (10 x 1).
  delta_3 = a3 - y_vec; 

%STEP 3: Compute error in hidden layer.
  %(26 x 1) = (26 x 10) * (10 x 1) * (26 x 1).
  delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)]; 
  
%STEP 4:Accumulate the gradient.
  delta_2 = delta_2(2:end);
  %(25 x 401) = (25 x 1) * (1 x 401).
  Theta1_grad += delta_2 * a1'; 
  % (10 x 26) = (10 x 1) * (1 x 26).
  Theta2_grad += delta_3 * a2';
  
end

row1 = size(Theta1_grad)(1);
row2 = size(Theta2_grad)(1);


Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(row1,1), Theta1_noBias];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(row2, 1), Theta2_noBias];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end