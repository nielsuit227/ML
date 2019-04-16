%% Start fresh
clear
close all
clc
warning off
%% Settings
% Kernel
sig = 2.5;
l = 5;
% Noise
sy = 0.0001;
% Confidence interval
conf = 25;
% Convergence tolerance
tol = 0.05;
% Expected Improvement variance
s = 1;
%% Kernel
K = @(X1,X2) sig^2*exp(-(X1-X2)'*(X1-X2)/(2*l^2));
%% (Unknown) objective function
obj = @(x) x^4 + 5*(x-3)^2 + 4*x^3 + 150*sin(1.5*x) + 10*randn;
fplot(obj)
hold on
%% Get two initials
X = zeros(4,1); y = zeros(4,1);
X(1) = 0;
X(2) = 1;
X(3) = 3;
X(4) = -3;
y(1) = obj(X(1));
y(2) = obj(X(2));
y(3) = obj(X(3));
y(4) = obj(X(4));
plot(X,y,'r*')
%% Interval
x1 = -4;
x2 = 4;
Xs = (x1:0.01:x2)';
n = length(Xs);
%% Gram matrix voor discrete interval
for i =1:n
    for j=1:n
        cov_ss(i,j) = K(Xs(i),Xs(i));
    end
end
%% Bayesian optimization
while(1)
    pause(2)
    hold off
    fplot(obj,[x1,x2])
    hold on
    plot(X,y,'r*')
    axis([-5 5 -200 500])
    drawnow
    m = length(X);
    % Start with covariance matrices
    for i=1:m
        for j=1:n
            cov_s(i,j) = K(X(i),Xs(j));
        end
        for j=1:m
            cov(i,j) = K(X(i),X(j));
        end
    end
    % Add measurement noise
    cov = cov + sy^2*eye(m);
    % Calculate joint variance
    CX = cov_ss - cov_s'/cov*cov_s;
    % Calculate joint mean
    MU = cov_s'/cov*y;
    % SVD (instead of Cholesky decomposition)
%     L = chol(CX+1e-6*eye(n),'lower');
    [U,S,~] = svd(CX);
    L = U*sqrt(S);
    SIG = conf*L*ones(n,1);
    plot(Xs,MU+SIG,'r')
    plot(Xs,MU-SIG,'b')
    axis([-5 5 -200 500])
    drawnow
    % Compute Expected Improvement
    fs = min(y);
    CDF = 1/2.*(1-erf((fs-MU)./SIG./sqrt(2)));
    PDF = 1/sqrt(2*pi)/s*exp(-1/2* ((fs-MU)./SIG).^2/s^2);
    EI = (fs - MU).*CDF + SIG.*PDF;
    pause(2)
    plot(Xs,EI)
    [EIm,i] = max(EI);
    pause(2)
    if EIm <= tol
        break
    end
    plot([Xs(i) Xs(i)],[-200,500],'k')
    X(m+1) = Xs(i);
    y(m+1) = obj(X(m+1));
end
    
    
    