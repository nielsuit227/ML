clear
close all
clc
%% Load and visualize data
load('Double_Gaussian_Large.mat')
%% K-means algorithm
% Preprocess
Y = X(:,2);
X = X(:,1);

%% Initial clusters
K = 10;
for n=1:K
    i = randi(length(X));
    j = randi(length(Y));
    Mu(n,:) = [X(i) Y(j)];
end
%%
plot(Mu(:,1),Mu(:,2),'kd')
C = zeros(1,length(X));
Z = C;
%%
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
        tx = [C==i]'.*X; tx(~any(tx,2),:) = [];
        ty = [C==i]'.*Y; ty(~any(ty,2),:) = [];
        plot(tx,ty,'o')
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
load('Double_Gaussian_Data.mat')
figure
gscatter(X(:,1),X(:,2),Y,'br')