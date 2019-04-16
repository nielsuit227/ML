%% Initializing data
clear
close all
clc
load('Lin_Reg_Data.mat') % 5x+6
x = x'; y=y';
%% Set prediction target
x_pred = 6;
y_real = 6*5+6
%% LLS
X = [x ones(size(x,1),1)];
t1 = (X'*X)\X'*y; % y=ax+b
y_lls = t1(1)*x_pred + t1(2)
%% GD
iterations = 10;
alpha = [0.01 0.1];
t2 =  ones(2,1);
m = length(y);
for n=1:iterations
    h = X * t2;
    t2(1) = t2(1) - alpha(1)/m*sum((h-y).*x);
    t2(2) = t2(2) - alpha(2)/m*sum(h-y);
% h = t1*X + t2, 1/2 sum (h-y)^2 equals therefore 1/2 sum (t1*X+t2-y)^2.
% Derivative to each variable is rather simple. Learning rate is slightly
% adjusted for the result to make sense. Otherwise more iterations are
% necessary. 
end
y_gd = t2(1)*x_pred + t2(2)
%% SGD
t3 = ones(2,1);
for i=1:iterations
    for ii = 1:m
        h = X(ii,:)*t3;
        t3(1) = t3(1) - alpha(1)*(h-y(ii))*X(ii,1);
        t3(2) = t3(2) - alpha(2)*(h-y(ii));
    end
end
y_sgd = t3(1)*x_pred + t3(2)
%% Visual Results
plot(x,y,'o')
hold on
plot(x,t1(1)*x+t1(2))
plot(x,t2(1)*x+t2(2))
plot(x,t3(1)*x+t3(2))
legend('data','LLS','GD','SGD')