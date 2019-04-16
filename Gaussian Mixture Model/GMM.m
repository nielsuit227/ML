%%% Gaussian Mixture Model using the Expectation Maximization (EM-)
%%% algorithm. K is parameterized for now but should be looped later on
%%% (grid search, hyperparameter)
clear
close all
clc
set(0,'DefaultLineLineWidth',2);            % Set linethickness for plots
load double_gaussian_data
% X(1001:end,:) = [];
X = (X-mean(X))./sqrt(var(X));              % Normalized data
%% PARAMETER - Gaussians
k = 2;
%% Initialize model parameters
[n,m] = size(X);
% Initialize means
mu = X(linspace(1,k,k)',:);
% Initialize classes
for i=1:n
    for j=1:k
        t(j) = norm(mu(j,:) - X(i,:));
    end
    [~,l] = min(t);
    Z(i) = l;
end
% Initialize phi
sig = zeros(m,m,k);
for j=1:k
    phi(j) = 1/n*sum(Z == j);
    for i=1:n
        sig(:,:,j) =sig(:,:,j) +  (Z(i) == j)/sum(Z==j) * (X(i,:) - mu(j,:))'* (X(i,:) - mu(j,:));
    end        
end    
%% Visualize initalization
figure
gscatter(X(:,1),X(:,2),Z)
hold on
plotgaussian(X(Z==1,:),mu(1,:),sig(:,:,1))
plotgaussian(X(Z==2,:),Y(Z==2),mu(2,:),sig(:,:,2))
plot(mu(:,1),mu(:,2),'k*')
%% EM algorithm
while(1)
    % E-step 
    for j = 1:k
        p(:,j) = diag(1/sqrt(det(2*pi*sig(:,:,j)))*exp(-1/2*((X-mu(j,:))/sig(:,:,j))*(X-mu(j,:))'));
    end
    w = p.*phi./sum(p*phi',2);
    % M-step
    sig = zeros(m,m,k);
    phi = 1/n * sum(w);
    mu_s = mu;
    mu = w'*X./sum(w)';    
    for j = 1:k
        sig(:,:,j) = (w(:,j).*(X-mu(j,:)))'*(X-mu(j,:))/sum(w(:,j));
    end
    %% Sole purpose of visualization
    [~,Z] = max(w');
    hold off
    plotgaussian(X,Z',mu(1,:),sig(:,:,1))
    hold on
    for j = 2:k
        plotgaussian(X,Z',mu(j,:),sig(:,:,j))
    end
    plot(mu(:,1),mu(:,2),'k*')
    drawnow
    pause(0.5)
    if norm(mu-mu_s) < 1e-3
        return
    end
end
    
%% Visualization function
function [] = plotgaussian(X,Z,Mu,Cov)
[n,m] = size(X);
if nargin < 4
    Cov = Mu;
    Mu = Z;
    Z = ones(n,1);
end
X0 = X - Mu;
[V D] = eig(Cov);
[D order] = sort(diag(D),'descend');
D = diag(D);
V = V(:,order);
t = linspace(0,2*pi,100);
e = [cos(t);sin(t)];
VV = 2*V*sqrt(D);
e = VV*e+Mu';
i = rand;
plot(e(1,:),e(2,:),'k')
hold on
gscatter(X(:,1),X(:,2),Z)
axis equal
legend off
end
