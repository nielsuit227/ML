clear
close all
clc
%% Parameters
T = 1000000;
mu = 50;
cov = 10;
%% Data generation
X = mvnrnd(mu,cov,T);
%% Central Limit Theorem
z = 1;
for i=1:T
    Z(i) = sqrt(i)*(sum(X(1:i))/i - mu);
    if i/T*100 > z
        clc
        fprintf('%.0f %% done',z)
        z = z + 1;
    end
end
%%
plot(Z)
title('$$\sqrt{T}(\hat{\mu} - \mu)$$','Interpreter','latex','FontSize',13)