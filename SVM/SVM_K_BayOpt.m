%% Start
clear
close all
clc
%% Bayesian Hyperparameter Optimization
% Input X is two dimensional and contains both hyperparameters, the
% regularization term C and the Kernel width sigma. The output is the
% empirical error plus the generalization error of a validation set.
% Settings
z = 7;          % Interval 
s = 1;       % Opt accuracy    
t = linspace(-z,3,(2*z+1)/s);
[U,V] = meshgrid(2.^t,2.^t);
Xs = [U(:),V(:)];
X = [Xs(1,:); Xs(211,:); Xs(end,:); Xs(15,:); Xs(145,:)];
% Optimization settings
% Make sure the optimization is at least run once with small length. This
% makes the algorithm very exploring instead of exploiting. 
sig = 5;
l = 4;
sy = 0.0001;
gp_tol = 0.05;
gp_conf = 2;
% Kernel
K_BO = @(X,Y) sig^2*exp(-(X-Y)'*(X-Y)/(2*l^2));
% Acquire initial
Y = zeros(5,1);
for i = 1:5
    Y(i) = runAlgo(X(i,1),X(i,2));
end
%% Visuals
figure
plot3(X(:,1),X(:,2),Y,'r*')
hold on
%% Gram matrix for sampling covariance
n = size(Xs,1);
for i=1:n
    for j=1:n
        cov_ss(i,j) = K_BO(Xs(i,:)',Xs(j,:)');
    end
end
%% Gaussian Process Regression
% Bayesion optimization using Expected Improvement. 
while(1)
    hold off
    plot3(X(:,1),X(:,2),Y,'k*')
    hold on
    plot3(X(end,1),X(end,2),Y(end),'r*')
    drawnow
    m = length(X);
    % Gram matrices for known data
    for i=1:m
        for j=1:n
            cov_s(i,j) = K_BO(X(i,:)',Xs(j,:)');
        end
        for j=1:m
            cov(i,j) = K_BO(X(i,:)',X(j,:)');
        end
    end
    % Add measurement noise
    cov = cov + sy*eye(m);
    % Joint variance
    CX = cov_ss - cov_s'/cov*cov_s;
    % Joint mean
    MU = cov_s'/cov*Y;
    % SVD of cov
    [U,S,~] = svd(CX);
    % Lower bound
    SIG = gp_conf*U*sqrt(S)*ones(n,1);
    % Minimum value as of now
    fs = min(Y);
    % Expected Improvement
    CDF = 1/2.*(1-erf((fs-MU)./SIG./sqrt(2)));
    PDF = 1/sqrt(2*pi)/s*exp(-1/2* ((fs-MU)./SIG).^2/s^2);
    EI = (fs - MU).*CDF + SIG.*PDF;
    [EIm,i] = max(EI);
    if EIm <= gp_tol
        fprintf('Maximum of Expected Improvement is zero\n')
        break
    end
    X(m+1,:) = Xs(i,:);
    Y(m+1) = runAlgo(Xs(i,1),Xs(i,2));
end
%% Process result
[Err,i] = min(Y);
fprintf('Optimal regularization term: %.4f\n',X(i,1))
fprintf('Optimal Kernel width       : %.4f\n',X(i,2))
fprintf('Generalization error       : %.2f %% \n',Err/3)
Visualize_Boundary(X(i,1),X(i,2))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% ALL NECESSARY FUNCTIONS %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = Visualize_Boundary(C,sig)
%% Load training data
X = []; Y = [];
load('SVM_NL_Data.mat')
%% Initialize Kernel function
K = @(X,Y) exp(-(X-Y)'*(X-Y)/(2*sig^2));
G = gram(K,X,X);
%% Initialize vectors
n = size(X,1);
a = zeros(n,1);
nznc = ones(n,1);
nc = 0;
exall = 1;
itmax = 1;
WG = 0;
c = 0;
E = ((a.*Y)'*G)' - Y;
%% Main loop (outer loop for first Lagrangian selection)
while((nc>0 || exall) && itmax)
    nc = 0;
    nznc = (a~=0).*(a~=C);
    if(exall)
        for i=1:n
            [a,E,ac] = ex(X,Y,i,a,G,C,E);
            nc = nc + ac;
        end
    else
        for i=nznc'.*linspace(1,100,100)
            if i == 0
                continue
            end                
            [a,E,ac] = ex(X,Y,i,a,G,C,E);
            nc = nc + ac;
        end
    end
    if exall == 1
        exall = 0;
    elseif nc == 0
        exall = 1;
    end
    c = c +1;
%     fprintf('Iteration %.0f, %.0f alphas changed\n',c,nc)
    if c>50 && WG == 0;
        warning('No quick convergence')
        WG = 1;
    end
    if c>500
        warning('Hyperparameters not converged')
        itmax = 0;
    end
end
%% Second loop (inner loop for second Lagrangian selection)
function [a,E,nc] = ex(X,Y,i,a,G,C,E)
nc = 0;
E = E;
a = a;
nznc = (a~=0).*(a~=C);
n = length(Y);
% KKT Conditions
Ri = (a.*Y)'*G(:,i);
if (a(i) == 0 && Ri*Y(i) < 1) || (a(i)<=C && Ri*Y(i) ~= 1) || (a(i) == C && Ri*Y(i) > 1)
    % Best possible step (based on error vector)
    [~,j] = min(sign(E(i)*E));
    [a,E,ac] = Update(X,Y,i,j,a,G,C);
    nc = nc + ac;
    for j=nznc'.*linspace(1,100,100)
        if j == 0
            continue
        end
        [a,E,ac] = Update(X,Y,i,j,a,G,C);
        nc = nc + ac;
    end
    for j=[1:i-1,i+1:n]
        [a,E,ac] = Update(X,Y,i,j,a,G,C);
        nc = nc + ac;
    end        
end
end
%% Update loop
function [a,E,ac] = Update(X,Y,i,j,a,G,C)
% Just function stuff
tol = 1e-3;
E = ((a.*Y)'*G)' - Y;
a=a;
ac = 0;
nznc = (a~=0).*(a~=C);
% Calculate errors
Ej = (a.*Y)'*G(:,j) - Y(j);
Ei = (a.*Y)'*G(:,i) - Y(i);
% Save data
a_ihold = a(i);
a_jhold = a(j);
% Calculate bounds
if Y(i) ~= Y(j)
    L = max(0,a(j)-a(i));
    H=min(C,C+a(j)-a(i));
else
    L=max(0,a(i)+a(j)-C);
    H=min(C,a(i)+a(j));
end
% if singular than skip
if L==H
    return
end
% second derivative obj fun
eta = 2*G(i,j) -G(j,j) - G(i,i);
% if positive we don't improve (Newton's method?)
if eta >= 0
    return
end
% New Lagrange Multiplier alpha_j
a(j) = a(j) - Y(j)*(Ei-Ej)/eta;
% Check for bounds
if a(j) > H
    a(j) = H;
elseif a(j) < L
    a(j) = L;
end
% Check for numerical inaccuracies while converged
if norm(a(j)-a_jhold) < tol
    return
end
% Update Lagrange Multiplier alpha_i
a(i)=a(i) + Y(i)*Y(j)*(a_jhold-a(j));
% Classification errors
E = ((a.*Y)'*G)' - Y;
ac = 1;
end
%% Gram matrix
function [G] = gram(K,X,Y)
n = size(X,1);
m = size(Y,1);
for i=1:n
    for j=1:m
        G(i,j) = K(X(i,:)',Y(j,:)');
    end
end
end
%% Visualize grid
Grid = linspace(1,100,100);
f = zeros(100,100);
for i=1:length(a)
    for j=Grid
        for k=Grid
            f(j,k) = f(j,k) + a(i)*Y(i)*K([j k]',X(i,:)');
        end
    end
end
hyp = f >= 0;
figure
imagesc(hyp);
hold on
gscatter(X(:,1),X(:,2),Y)
end
%% Complete SVM function (using SMO)
function [Err] = runAlgo(C,sig)
%% Load training data
X = []; Y = [];
load('SVM_NL_Data.mat')
%% Initialize Kernel function
K = @(X,Y) exp(-(X-Y)'*(X-Y)/(2*sig^2));
G = gram(K,X,X);
%% Initialize vectors
n = size(X,1);
a = zeros(n,1);
nznc = ones(n,1);
nc = 0;
exall = 1;
itmax = 1;
WG = 0;
c = 0;
E = ((a.*Y)'*G)' - Y;
%% Main loop (outer loop for first Lagrangian selection)
while((nc>0 || exall) && itmax)
    nc = 0;
    nznc = (a~=0).*(a~=C);
    if(exall)
        for i=1:n
            [a,E,ac] = ex(X,Y,i,a,G,C,E);
            nc = nc + ac;
        end
    else
        for i=nznc'.*linspace(1,100,100)
            if i == 0
                continue
            end                
            [a,E,ac] = ex(X,Y,i,a,G,C,E);
            nc = nc + ac;
        end
    end
    if exall == 1
        exall = 0;
    elseif nc == 0
        exall = 1;
    end
    c = c +1;
%     fprintf('Iteration %.0f, %.0f alphas changed\n',c,nc)
    if c>50 && WG == 0;
        warning('No quick convergence')
        WG = 1;
    end
    if c>500
        warning('Hyperparameters not converged')
        itmax = 0;
    end
end
%% Second loop (inner loop for second Lagrangian selection)
function [a,E,nc] = ex(X,Y,i,a,G,C,E)
nc = 0;
E = E;
a = a;
nznc = (a~=0).*(a~=C);
n = length(Y);
% KKT Conditions
Ri = (a.*Y)'*G(:,i);
if (a(i) == 0 && Ri*Y(i) < 1) || (a(i)<=C && Ri*Y(i) ~= 1) || (a(i) == C && Ri*Y(i) > 1)
    % Best possible step (based on error vector)
    [~,j] = min(sign(E(i)*E));
    [a,E,ac] = Update(X,Y,i,j,a,G,C);
    nc = nc + ac;
    for j=nznc'.*linspace(1,100,100)
        if j == 0
            continue
        end
        [a,E,ac] = Update(X,Y,i,j,a,G,C);
        nc = nc + ac;
    end
    for j=[1:i-1,i+1:n]
        [a,E,ac] = Update(X,Y,i,j,a,G,C);
        nc = nc + ac;
    end        
end
end
%% Update loop
function [a,E,ac] = Update(X,Y,i,j,a,G,C)
% Just function stuff
tol = 1e-3;
E = ((a.*Y)'*G)' - Y;
a=a;
ac = 0;
nznc = (a~=0).*(a~=C);
% Calculate errors
Ej = (a.*Y)'*G(:,j) - Y(j);
Ei = (a.*Y)'*G(:,i) - Y(i);
% Save data
a_ihold = a(i);
a_jhold = a(j);
% Calculate bounds
if Y(i) ~= Y(j)
    L = max(0,a(j)-a(i));
    H=min(C,C+a(j)-a(i));
else
    L=max(0,a(i)+a(j)-C);
    H=min(C,a(i)+a(j));
end
% if singular than skip
if L==H
    return
end
% second derivative obj fun
eta = 2*G(i,j) -G(j,j) - G(i,i);
% if positive we don't improve (Newton's method?)
if eta >= 0
    return
end
% New Lagrange Multiplier alpha_j
a(j) = a(j) - Y(j)*(Ei-Ej)/eta;
% Check for bounds
if a(j) > H
    a(j) = H;
elseif a(j) < L
    a(j) = L;
end
% Check for numerical inaccuracies while converged
if norm(a(j)-a_jhold) < tol
    return
end
% Update Lagrange Multiplier alpha_i
a(i)=a(i) + Y(i)*Y(j)*(a_jhold-a(j));
% Classification errors
E = ((a.*Y)'*G)' - Y;
ac = 1;
end
%% Gram matrix
function [G] = gram(K,X,Y)
n = size(X,1);
m = size(Y,1);
for i=1:n
    for j=1:m
        G(i,j) = K(X(i,:)',Y(j,:)');
    end
end
end

%% Calculate Empirical error and return it
Emp_err = sum(Y ~= sign((a.*Y)'*G)');
%% Calculate Generalization error and return it
Yt = Y; Xt = X;
load('SVM_Val.mat')
G = gram(K,Xt,X);
Gen_err = sum(Y ~= sign((a.*Yt)'*G)');
%% Sum both
Err = Emp_err + Gen_err;
end       



