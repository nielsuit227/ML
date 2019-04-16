%% Initialization 
clear
close all
clc
load('Log_Reg_Data.mat')
%% Prediction 1/1+exp(-0.0484*(x-52.8905))
x_pred = 40;
%% Gradient Descent
iterations = 150;
alpha = [0.01,1];
t1 = [0.3;-30];                 % Define operating variables, one dimensional second order
x = linspace(1,100,100)';        % Define Estimation Space
close all
figure
plot(Y)
hold on
m = length(Y);
for n=1:iterations
    h_gd = 1./(1+exp(-t1(1)*(x+t1(2))));
    t1(1) = t1(1) + alpha(1)/m*sum((Y-h_gd).*x);
    t1(2) = t1(2) + alpha(2)/m*sum(Y-h_gd);
end
%% Newton's method
t2 = zeros(2,iterations); t2 = [0.3;-30];
alpha = [0.1;0.1];
x = [x ones(100,1)];
for n=1:iterations
    h_nm = 1./(1+exp(-t2(1,n)*(x(:,1)+t2(2,n))));
    diff_J = 1/m*sum((h_nm-Y).*x)';
    H1_J = 1/m*sum(h_nm.*(1-h_nm))*x'*x;
    t2(:,n+1) = t2(:,n)-alpha.*H1_J\diff_J;
end
%% True line
h_true = 1./(1+exp(-0.0484*(x(:,1)-52.8905)));
%% Visual Results
plot(h_gd)
plot(h_nm)
plot(h_true)
legend('Data','Grad Desc','Newton','True Probability')
