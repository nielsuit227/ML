function [X, sig] = GMMS_I(X,k)
% Gaussian Mixture Model Selection, selects k datapoints that represent the
% datapoints best. 
%
% Input         X       Data features
%               Y       Data labels
%               k       Fitted Gaussians
%
% Output        X       Feature representations
%               Y       Representation labels
%               sig     Covariance of representation
%% Initialization
fprintf('Finding best data representation\n\n')
global vis
if isempty(vis) 
    vis = 0
end

[n,m] = size(X);        % Input size
sig = eye(m)/20;         % Constraint covariance matrix
if vis
    figure
    title('Determining optimal data representation')
end
%% GMM init
tic;
iter = 1;
mu = X(linspace(1,k,k)',:);     % Initial means
for i = 1:n
    for j = 1:k
        t(j) = norm(mu(j,:)-X(i,:));
    end
    [~,l] = min(t);
    Z(i) = l;
end
for j = 1:k
    phi(j) = 1/n*sum(Z==j);
end
init_time = toc;
%% EM Algorithm 
while(1)
    fprintf('Gaussian Mixture Model - iteration %.0f\n', iter)
    iter = iter + 1;
    % E-step - Calculates probabilities of clusters
    tic;
    for j = 1:k
        p(:,j) = diag(det(2*pi*sig)^(-1/2)*exp(-1/2*((X-mu(j,:))/sig)*(X-mu(j,:))'));
    end
    w = p.*phi./(p*phi');
    prob_t = toc;
    % M-step - Updates cluster parameters
    tic;
    phi = 1/n * sum(w);
    mu_s = mu;
    mu = w'*X./sum(w)';    
    mstep_t = toc;
    % Stopping criterion - (Cluster mean didn't change)
    if norm(mu_s-mu) < 2*k*1e-3
        break
    end
    % Not converged
    if any(isnan(phi))
        error('Gaussian Mixture Model became singular, enlarge sigma')
        break
    end
end
fprintf('\n\n *** Gaussian Mixture Model - CONVERGED ***\n\n')
%% VISUALIZATION
fprintf('Plotting representation with covariance\n')
if vis
    [~,Z] = max(w');
    hold off
    plotgaussian(X,Z',mu(1,:),sig);
    for j = 2:k
        plotgaussian(X,Z',mu(j,:),sig);
    end
    plot(mu(:,1),mu(:,2),'k*')
    legend off
    drawnow
    pause(0.5)
end
X = mu;
end

