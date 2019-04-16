%%% Partial Least Squares. Similar to PCA but fits a regression model
%%% instead of hyperplanes on which the data is projected. This case
%%% labelled data is fitted by only 3 parameters. 
clear
close all
clc
load('PLS.mat')
%% Settings
l = 1;
%% Algorithm
W = []; P = []; Q = [];
Xk = x - ones(size(x,1),1)*mean(x);
wk = Xk'*y / norm(Xk'*y);
for k=1:l
    tk = Xk*wk;
    tt = tk'*tk;
    tk = tk/tt;
    p = Xk'*tk;
    q = y'*tk;
    if q == 0
        break
    end
    if k<l
        Xk = Xk - tt*tk*p';
        wk = Xk'*y;
    end
    W = [W wk];
    P = [P p];
    Q = [Q;q];
end
        
B = W/(P'*W)*Q;


figure
plot(x*B)
hold on
plot(y)
plot([0, 101],[0.5 0.5])
legend('Estimation','Data','Treshold')