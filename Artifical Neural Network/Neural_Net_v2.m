%%% Changelog V2: 
%%% Epochs and learning rate adapted to matlab example. 
%%% Sequentual Gradient Descent (instead of batch)

%% Clear and Load data
clear
close all
clc
[A,C] = mendez;
%% Network Size
ni = size(A,2);             % Input units
nh = 19;                    % Hidden units
ny = 1;                     % Output units
ne = 5000;                   % Epochs
m = size(A,1);              % Training examples
%% Activation function
f = @(x) 1./(1+exp(-x));      % I/O mapping by Log. Reg. (Sigmoid)
df = @(x) f(x).*(1-f(x));     % Plus its derivative df
%% Cost function and derivative
L = 1e-6;
J = @(W1,b1,W2,b2,x,y) 1/2*(f(W2*f(W1*x+b1)+b2) - y)^2 + L/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));
%% Learning rate and Regularization
alpha = 1;
%% Initializing Network
W1 = rand(nh,ni);    
W2 = rand(ny,nh);
b1 = rand(nh,1);
b2 = rand;
%% Network training
figure; hold on;
i=0;
while i < ne
    i = i+1;
    for ii = 1:m
        % Input/output from sample
        a1 = A(ii,:)';
        y1 = C(ii);
        %% Forward propogation
        z2 = W1*a1+b1;
        a2 = f(z2);
        z3 = W2*a2+b2;
        a3 = f(z3);
        %% Backward propogation
        d3 = -(y1-a3).*df(z3);       %% For input 1, layer 3
        d2 = (W2'*d3).*df(z2);
        %% Updating Weights (Seq. Grad. Desc.)
        b1 = b1 - alpha.*d2;
        b2 = b2 - alpha.*d3;
        W1 = W1 - alpha.*(d2*a1'+L*W1);
        W2 = W2 - alpha.*(d3*a2'+L*W2);
        %% Derivative check
%         eps = 1e-4;
%         n = 1;  m = 18;
%         alg_der_t = d3*a2'+L*W2; alg_der = alg_der_t(n,m)
%         W2p = W2; W2p(n,m) = W2(n,m) + eps; W2n = W2; W2n(n,m) = W2(n,m) - eps;        
%         num_der = (J(W1,b1,W2p,b2,a1,y1) - J(W1,b1,W2n,b2,a1,y1))/2/eps
    end
    
    % Cost function evaluation
    J =0;
    for ii=1:m
        y = C(ii);
        x = A(ii,:)';
        J = J + 1/2*(f(W2*f(W1*x+b1)+b2) - y)^2 + L/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));
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
    
        
        
        
    

