%%% Principle Component Analysis. Optimizes a hyperplane on which the data
%%% is projected for maximum variance on that hyperplane. Does so by a
%%% simple SVD and projecting on the D principle components. 
clear
close all
clc
%% Load and visualize data
load Double_Gaussian_Data
[n,m] = size(X);
X = (X - ones(n,1)*mean(X))./sqrt(var(X));
%% Singular Value Decomposition
%%%%%%%%%%%%%%%%%%%%%%%%%%
D = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,~,V] = svd(X);
v_svd = V(:,1);
X_svd = (v_svd'*X')';
%% Eigenvalue Decomposition
Xe = X';
[U,D] = eig(Xe*Xe');
[i,j] = sort(-diag(D));
v_ed = U(:,j(1));
X_ed = (v_ed'*Xe)';
%% Plot everything
plot([0, v_svd(1)], [0, v_svd(2)])
hold on
plot([0, v_ed(1)], [0, v_ed(2)])
gscatter(X(:,1),X(:,2),Y,'br')
figure
subplot(1,2,1)
gscatter(X(:,1),X(:,2),Y,'br')
subplot(1,2,2)
gscatter(X_svd,ones(n,1),Y,'br')
hold on
gscatter(X_ed,-ones(n,1),Y,'br')
fprintf('Eigenvalue variance:       %.3f\n', var(X_ed))
fprintf('Singular Value variance:   %.3f\n', var(X_svd))










