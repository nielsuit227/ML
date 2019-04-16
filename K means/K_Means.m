clear
close all
clc
%% Data
K = 15;          % How many Clusters
Var = 10;        % Variance over clusters
Dis = 100;       % Unknown centroid distances
X = [];
Y = [];
for n=1:K
    X = [X;randi(Dis)*ones(10,1)+randi(Var,10,1)];
    Y = [Y;randi(Dis)*ones(10,1)+randi(Var,10,1)];
end
plot(X,Y,'o')
hold on
%% K-Means
K = input('How many Clusters\n');
for n=1:K
    i = randi(length(X));
    j = randi(length(Y));
    Mu(n,:) = [X(i) Y(j)];
end
plot(Mu(:,1),Mu(:,2),'R*')
C = zeros(1,length(X));
Z = C;
for n=1:15
    close all
    for i=1:length(X)
        t = [];
        for ii=1:K 
            t = [t; sqrt((Mu(ii,1)-X(i))^2+(Mu(ii,2)-Y(i))^2)];
        end
        [~,C(i)] = min(t);
    end
    for i=1:K
        Mu(i,:) = [sum([C==i]'.*X)/sum(C==i),sum([C==i]'.*Y)/sum(C==i)];
        hold on
        plot([C==i]'.*X,[C==i]'.*Y,'o')
    end
    hold on
    plot(Mu(:,1),Mu(:,2),'R*')
    pause(1)
    if C == Z
        fprintf('Converged\n')
        break
    end
    Z = C;
end
