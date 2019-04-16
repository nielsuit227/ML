%%% Near mean classifier. L1 loss function with L1 regularization. 
clear
close all
clc
load digits
global vis
vis = 1;
%% Initalization
[n,m] = size(X);n=0.5*n;
Xp = X(1:n,:);
Xn = X(n+1:end,:);
mp = zeros(1,m);%mean(Xp);
mn = zeros(1,m);%mean(Xn);
a = 0;
R = 0;
L = 0.1;
tol = 1e-3;
%% Gradient Descent
i=1;
while(1)
    cache.mp = mp;
    cache.mn = mn;
    mp = mp - L/n*( -sum((Xp-mp)) + R*(mp-mn) );
    mn = mn - L/n*( sum((Xn+mn)) - R*(mp-mn) );
    if norm(cache.mp-mp) < tol && norm(cache.mn-mn) < tol
        fprintf('Gradient Descent converged\n')
        return
    end
    fprintf('Iteration %.0f\n',i)
    i = i+1;
    if vis
        hold off
        plot(mn)
        hold on
        plot(mp)
        legend('Negative','Positive')
        drawnow
    end
end
%%
plot(mean(Xp))
plot(mean(Xn))
legend('Negative','Positive','Mean positive','Mean negative')

    