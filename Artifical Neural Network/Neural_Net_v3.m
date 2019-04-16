%%% Changelog V3:
%%% Extra 2 layers added to see sensitivity

%% Clear and Load data
clear
close all
clc
[A,C] = mendez;
%% Network Size
ni = size(A,2);             % Input units
nh = 19;                    % Hidden units
ny = 1;                     % Output units
ne = 10000;                   % Epochs
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
W2 = rand(nh,nh);
W3 = rand(nh,nh);
W4 = rand(ny,nh);
b1 = rand(nh,1);
b2 = rand(nh,1);
b3 = rand(nh,1);
b4 = rand;
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
        z4 = W3*a3+b3;
        a4 = f(z4);
        z5 = W4*a4+b4;
        a5 = f(z5);
        %% Backward propogation
        d5 = -(y1-a5).*df(z5);
        d4 = (W4'*d5).*df(z4);
        d3 = (W3'*d4).*df(z3);       
        d2 = (W2'*d3).*df(z2);
        %% Updating Weights (Seq. Grad. Desc.)
        b1 = b1 - alpha.*d2;
        b2 = b2 - alpha.*d3;
        b3 = b3 - alpha.*d4;
        b4 = b4 - alpha.*d5;
        W1 = W1 - alpha.*(d2*a1'+L*W1);
        W2 = W2 - alpha.*(d3*a2'+L*W2);
        W3 = W3 - alpha.*(d4*a3'+L*W3);
        W4 = W4 - alpha.*(d5*a4'+L*W4);
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
        J = J + 1/2*(f(W4*f(W3*f(W2*f(W1*x+b1)+b2)+b3)+b4) - y)^2 + L/2*(sum(sum(W1.^2))+sum(sum(W2.^2))+sum(sum(W3.^2))+sum(sum(W4.^2)));
    end
    plot(i,J/m,'*')
end
%% Fit function
Y_es = @(x) f(W4*f(W3*f(W2*f(W1*x+b1)+b2)+b3)+b4);

for i=1:m
    Es(i) = Y_es(A(i,:)');
end
figure
plot(C)
hold on
plot(Es)
    
        
        
        
    

