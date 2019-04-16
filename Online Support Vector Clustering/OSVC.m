%%% Script after paper[1]. Clustering algorithm which uses Support Vector
%%% Description. Implemented in online manner using Regularized Follow The
%%% Leader with negative entropy regularizer.
%%% http://www.jmlr.org/papers/volume2/horn01a/horn01a.pdf 
clear
close all
clc
load Moons                          % Load data
% moons - C=350, sigma 0.19
X = (X-mean(X))./sqrt(var(X));      % Normalize data
[n, m] = size(X);                   % Note size
%% Select multipliers - GMM
global vis
vis = 0;
th = 0.3;                           % Threshold for outliers (to encorporate into data representation)
sel_data = 1000;                    % Determine selection data
N = 50;                             % Number of representations
[XC, sig] = GMMS_I(X(1:sel_data, :), N);   % Fit using Gaussian Mixture Model
%% Visualize data
figure
gscatter(X(:,1),X(:,2),ones(n,1),'ro')
hold on
gscatter(XC(:,1),XC(:,2),ones(N,1),'ko')
%% Parameters
algo = 'OGD';
epoch = 1; ep=0;
C = 350;
sigma = 0.19;
eta = 0.005;
K = @(X,Y) 1/sqrt(2*pi*sigma)*exp(-diag((X-Y)*(X-Y)')/(2*sigma^2));
%% Initializing
grad = zeros(N, 1);
s_grad = zeros(N, 1);
alpha = zeros(N, 1);
alpha2 = alpha;
% Online algo
mem.a = zeros(epoch*n,1);
mem.out = XC;
while ep < epoch
    for t = 1:n
        % Update multipliers
        switch algo
            case 'OGD'
                % Update multipliers - OGD
                grad = K(XC, X(t, :));
                alpha = alpha + eta*grad;
                if sum(alpha)>C/2
                    alpha = alpha/sum(alpha)*C/2;
                end
            case 'RFTL'
                % Update multipliers - RFTL
                s_grad = s_grad + grad;
                if sum(s_grad) > (log(C)+1)/eta
                    s_grad = s_grad/sum(s_grad)*(log(100)+1)/eta;
                end
                alpha2 = exp(-1+eta*s_grad);
        end
        mem.a(t+ep*n) = alpha(50);
        % Check outliers
        p = diag(exp(-1/2*(X(t,:)-XC)*(X(t,:)-XC)'));
        if ~any(p>th)
            mem.out = [mem.out; X(t,:)];
            if size(mem.out,1) > sel_data
                [XC,sig] = GMM_I(XC,N);
                mem.out = XC;
            end
        end
        fprintf('[%.2f %%] Iteration %.0f done - %.0f outliers\n',100*(ep*n+t)/epoch/n, t+ep*n, size(mem.out,1)-N)
    end
    ep = ep + 1;
end
%% Cluster assignment
fprintf('\n\n Determine Clusters \n\n')
EQ = [];
for i = 1:N
    Xeq = XC(i,:);
    while(1)
        store = Xeq;
        Xeq = sum(alpha.*K(XC, Xeq).*XC,1)/sum(alpha.*K(XC, Xeq));
        if norm(Xeq - store,2) < 1e-2
            EQ = [EQ; round(Xeq,1)];
            break
        end
    end
end
fprintf('Determined equilibrium points\n')
EQ = unique(sort(round(EQ, 1), 2), 'rows');
steps = 20;
fprintf('Determine adjacency matrix (%.0f steps)\n', steps)
q = size(EQ,1);
A = ones(q);
for i = 1:q
    for j = i:q
        for k = 1:steps
            intermediate_step = (k-1)/(steps-1)*EQ(i,:) + (steps-k)/(steps-1)*EQ(j,:);
            F = alpha'*K(XC, intermediate_step);
            if F < 1
                A(i,j) = 0;
            end
        end
    end
end
A = floor((A+A')/2);
r = fliplr(symrcm(A));
% Get the clusters
C = zeros(q,1);
cc = 1;             % Clustercount
for i = 1:q
    if C(i) == 0
        if any(A(:,i) == 1)
            C(A(:,i)==1) = cc;
            cc = cc+1;
        end
    else
        if any(A(:,i) == 1)
            C(A(:,i)==1) = C(i);
        end
    end
end
%% Convergence
figure
plot(mem.a)
title('Convergence multiplier 50')
%% Visualize field
eps = 1;
z = 100;
xmi = 1.1*min(min(X));
xma = 1.1*max(max(X));
Grid = linspace(xmi, xma, z);
f = zeros(z, z);
for i =1:N
    for j = 1:z
        for k = 1:z
            f(j, k) = f(j, k) + alpha(i)*K([Grid(j) Grid(k)], XC(i, :));
        end
    end
end
figure
surf(f)
title('Kernel PDF / 3D decision surface OGD-SVC on toy dataset')
figure
mmin = abs(min(min(f-1))); mmax = max(max(f-1)); cmin = floor(50*mmin/(mmin+mmax)); cmax = ceil(50*mmax/(mmin+mmax));
cmap(1:cmin,:) = [linspace(0,1,cmin)' linspace(0,1,cmin)' ones(cmin,1)];
cmap(1:cmin,:) = [ones(cmin,1) ones(cmin,1) ones(cmin,1)];
cmap(cmin+1:cmin+cmax,:) = [ones(cmax,1) linspace(1,0,cmax)' linspace(1,0,cmax)'];
image(Grid,Grid,f-1,'CDataMapping','scaled')
hold on
colormap(cmap)
contour(Grid,Grid,f-1,[0 0],'k-')
contour(Grid,Grid,f-1,[eps eps],'r--')
contour(Grid,Grid,f,[0 0],'b-')
plot_n = 1000;
gscatter(X(1:plot_n,2),X(1:plot_n,1),ones(plot_n,1),'bo')
gscatter(XC(:,2),XC(:,1),ones(N,1),'ko')
gscatter(EQ(:,2),EQ(:,1),ones(q,1),'m^')
legend off
title('Cluster Contour OGD-SVC on toy dataset')



% f = zeros(z, z);
% for i =1:N
%     for j = 1:z
%         for k = 1:z
%             f(j, k) = f(j, k) + alpha2(i)*K([Grid(j) Grid(k)], XC(i, :));
%         end
%     end
% end
% figure
% surf(f)
% figure
% mmin = abs(min(min(f-1))); mmax = max(max(f-1)); cmin = floor(50*mmin/(mmin+mmax)); cmax = ceil(50*mmax/(mmin+mmax));
% cmap(1:cmin,:) = [linspace(0,1,cmin)' linspace(0,1,cmin)' ones(cmin,1)];
% cmap(1:cmin,:) = [ones(cmin,1) ones(cmin,1) ones(cmin,1)];
% cmap(cmin+1:cmin+cmax,:) = [ones(cmax,1) linspace(1,0,cmax)' linspace(1,0,cmax)'];
% image(Grid,Grid,f-1,'CDataMapping','scaled')
% hold on
% colormap(cmap)
% contour(Grid,Grid,f-1,[0 0],'k-')
% contour(Grid,Grid,f-1,[eps eps],'r--')
% plot_n = 1000;
% gscatter(X(1:plot_n,2),X(1:plot_n,1),ones(plot_n,1),'bo')
% gscatter(XC(:,2),XC(:,1),ones(N,1),'ko')
% title('Regularized Follow The Leader')

