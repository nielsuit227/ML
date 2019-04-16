clear 
close all
clc
%% Function 
Y = @(x) x.^2;
figure
fplot(Y,[-100,100])
title('Values of cost function')
hold on
%% Grad Descent
x = zeros(1,100);
x(1) = 75;
a = 0.1;
for n=2:100
    x(n) = x(n-1) - a*2*x(n-1); %% - der.
    plot(x(n),Y(x(n)),'r*')
    pause(1)
    if x(n) <= 10
        axis([-10 10 0 100])
    end
    if x(n) <= 1
        axis([-1 1 0 2])
    end        
end
%% 
y = Y(x);
figure
plot(y)
title('Convergence')
