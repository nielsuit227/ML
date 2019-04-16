%%% Linear Discriminant Analysis. Supervised dimension reduction. Maximizes
%%% the covariance between in and output data by fitting two Gaussian's and
%%% projecting them. 
clear
close all
clc
load bankadd% Double_Gaussian_Data
X = Xval; Y=Yval;
figure
subplot(1,2,1)
gscatter(X(:,1),X(:,2),Y,'br')
%% Normalize X
[n,m] = size(X);
X = (X - ones(n,1)*mean(X))./sqrt(var(X));
%% Initialize
Mu1 = ones(size(X,2),1); 
Mu0 = ones(size(X,2),1); 
S_w = zeros(size(X,2));
n1 = sum(Y);n0 = length(Y) - sum(Y);
%% LDA
for n=1:length(Y)
    if Y(n) == 1
        Mu1 = Mu1 + X(n,:)'/n1;
    else
        Mu0 = Mu0 + X(n,:)'/n0;
    end
end
for n = 1:length(Y)
    if Y(n) == 1
        S_w = S_w + (X(n,:)' - Mu1)*(X(n,:)'-Mu1)';
    else
        S_w = S_w + (X(n,:)' - Mu0)*(X(n,:)'-Mu0)';
    end
end
Mu = mean(X);
S_b = n1*(Mu1-Mu)*(Mu1-Mu)' + n0*(Mu0-Mu)*(Mu0-Mu)';
[e,v] = eig(S_w\S_b); v = diag(v);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = 2;  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
G = e(:,1:D);
X_r = X*G;
%%
subplot(1,2,2)
gscatter(X_r,ones(n,1),Y, 'br')