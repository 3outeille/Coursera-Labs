function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:1%num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    %For loop version.
    %=================
    %store_theta = [];
    %Compute theta.
    %for j = 1:size(X,2)
    %  val = (X*theta - y).*X(:, j);
    %  dJ = (1/m)*sum(val);
    %  tmp = theta(j) - alpha*dJ;
    %  Store theta
    %  store_theta = [store_theta; tmp];
    %endfor
    %Update theta similtaneously.
    %theta = store_theta;
    
    %Vectorization version.
    %======================
    val = (X*theta - y).*X;
    dJ = (1/m)*sum(val);
    theta = theta - alpha*dJ';
    
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
