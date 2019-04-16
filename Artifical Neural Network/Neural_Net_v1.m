%%% Code for a Single Layer Feed Forward Artificial Neural Network. Weight
%%% updates using Back Propogation Batch Gradient Descent. Activation function 
%%% is changeable but %%% Sigmoid function is chosen. Layer size is adjustable. 
%%% Dataset from a Matlab example:
%%% https://www.mathworks.com/matlabcentral/fileexchange/22010-a-very-simple-and-intuitive-neural-network-implementation
%%% Notation according to Andrew Ng:
%%% http://ufldl.stanford.edu/wiki/index.php/Neural_Networks

%% Clear and Load data
clear
close all
clc
[A,C] = mendez;
%% Network Size
ni = size(A,2);             % Input units
nh = 50;                    % Hidden units
ny = 1;                     % Output units
ne = 1000;                   % Epochs
m = size(A,1);              % Training examples
%% Activation function
f = @(x) 1./(1+exp(-x));      % I/O mapping by Log. Reg. (Sigmoid)
df = @(x) f(x).*(1-f(x));     % Plus its derivative df
%% Cost function and derivative
L = 0;
J = @(W1,b1,W2,b2,x,y) norm(f(W2*f(W1*x+b1)+b2) - y) + L/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));
%% Learning rate and Regularization
alpha = 0.5;
%% Initializing Network
W1 = zeros(nh,ni);    
W2 = zeros(ny,nh);
b1 = zeros(nh,1);
b2 = zeros;
%% Network training
figure; hold on;
i=0;
while i < ne
    i = i+1;
    dW1 = zeros(size(W1));
    dW2 = zeros(size(W2));
    db1 = zeros(size(b1));
    db2 = zeros(size(b2));
    for ii = 1:m
        % Input/output from sample
        a1 = A(ii,:)';
        y1 = C(ii);
        % Forward propogation through hidden and output layer to calculate
        % estimate
        z2 = W1*a1+b1;
        a2 = f(z2);
        z3 = W2*a2+b2;
        a3 = f(z3);
        % Calculate error derivatives
        d3 = -(y1-a3).*df(z3);       %% For input 1, layer 3
        d2 = (W2'*d3).*df(z2);
        % Accumulate error
        dW1 = dW1 + d2*a1';
        dW2 = dW2 + d3*a2';
        db1 = db1 + d2;
        db2 = db2 + d3;
    end
    % Gradient Descent
    W1 = W1 - alpha.*(dW1/m+L*W1);
    W2 = W2 - alpha.*(dW2/m+L*W2);
    b1 = b1 - alpha.*db1/m;
    b2 = b2 - alpha.*db2/m;
    
    % Cost function evaluation
    J =0;
    for ii=1:m
        y = C(ii);
        x = A(ii,:)';
        J = J + norm(f(W2*f(W1*x+b1)+b2) - y) + L/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));
    end
    plot(i,J/m,'*')
end
%% Fit function
Y_es = @(x) f(W2*f(W1*x+b1)+b2);

for i=1:m
    Es(i) = Y_es(A(i,:)');
end
figure
plot(C)
hold on
plot(Es)
    
        
        
        
    

