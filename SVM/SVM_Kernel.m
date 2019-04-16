clear 
close all
clc
str = 'SVM_NL_Data.mat';
load(str)
% yalmiptest
% gurobi_setup
% X = X/10;
%% Plot data
n = length(X(:,1));5
%% Settings
tol = 1e-3;
sigma = 20;    % Gaussian kernel width
C = 0.5;          % Regularization
%% Kernel function
K = @(x1,x2) exp(-norm(x1 - x2)^2/(2*sigma^2));
for i =1:n
    for j =1:n
        G(i,j) = K(X(i,:)',X(j,:)');
    end
end
%% Initializing vectors
alpha = zeros(n,1);     % Lagrangian parameters
a = alpha;
z = 0;                  % Counter
b = 0;                  % Bias
%% Sequential Minimal Optimization
while (1) % as long as KKT aren't fulfilled keep running
    changed_alphas = 0;
    n = length(X(:,1));
    for i=1:n
    %% Check error
        Ei = (alpha.*Y)'*G(:,i) + b;
        %% KKT Conditions
        if (alpha(i) == 0 && Ei*Y(i) < 1) || (alpha(i)<=C && Ei*Y(i) ~= 1) || (alpha(i) == C && Ei*Y(i) > 1) %% (Y(i)*Ei < -tol && alpha(i) < C) || (Y(i)*Ei > tol && alpha(i) > 0)%%
            
            % if not fulfilled, run for entire training group except i
            for j=[1:i-1,i+1:n]
                % Error of match
            Ei = (alpha.*Y)'*G(:,i) + b - Y(i);    
            Ej = (alpha.*Y)'*G(:,j) + b - Y(j);
            % Save data
            a_ihold = alpha(i);
            a_jhold = alpha(j);
            % calculate bounds
            if Y(i) ~= Y(j)
                L = max(0,alpha(j)-alpha(i));
                H=min(C,C+alpha(j)-alpha(i));
            else
                L=max(0,alpha(i)+alpha(j)-C);
                H=min(C,alpha(i)+alpha(j));
            end
            % if singular than skip
            if L==H
                continue
            end
            % second derivative obj fun
            eta = 2*G(i,j) -G(j,j) - G(i,i);
            % if positive we don't improve (Newton's method?)
            if eta >= 0
                continue
            end
            %% New Lagrange Multiplier alpha_j
            alpha(j) = alpha(j) - Y(j)*(Ei-Ej)/eta;
            % Check for bounds
            if alpha(j) > H
                alpha(j) = H;
            elseif alpha(j) < L
                alpha(j) = L;
            end
            % Check for numerical inaccuracies while converged
            if norm(alpha(j)-a_jhold) < tol
                continue
            end
            %% Update Lagrange Multiplier alpha_i
            alpha(i)=alpha(i) + Y(i)*Y(j)*(a_jhold-alpha(j));
%             b = - (max((alpha.*Y)'*G*(Y==-1))+min((alpha.*Y)'*G*(Y==1)))/2;
%             b = - (max((alpha.*Y)'*G) + min((alpha.*Y)'*G))/2;
            b = 0;
            changed_alphas=changed_alphas+1;
            end
        end            
    end
    z = z+1;
    fprintf('Iteration %.0f done, %.0f alphas changed \n',z,changed_alphas)
    if changed_alphas==0
        % should reiterate a couple times
        break
    end
    X=X((find(alpha~=0)),:);
    Y=Y((find(alpha~=0)),:);
    alpha=alpha((find(alpha~=0)),:);
    G = G((find(alpha~=0)),find(alpha~=0));
end
%% Check KKT
for i=1:n
    if alpha(i) <= tol && Y(i)*((alpha.*Y)'*G(:,i)+b) < 1 || alpha(i) >= C-tol &&  Y(i)*((alpha.*Y)'*G(:,i)+b) > 1 || alpha(i) <= C-tol && alpha(i) >= tol &&  Y(i)*((alpha.*Y)'*G(:,i)+b) ~= 1
        warning('KKT Conditions not fulfilled, no optimal primal')
        break
    end
end
%% Determine error
% Yp_t = (a.*Y)'*G;
%% Discretisize boundary
visualize(X, Y, alpha, b, K, X, Y)

function [] = visualize(XC,YC,alpha,b,K,X,Y)
n = 100;                            % Grid size
xmi = 1.1*min(min(X));             % Minimum X
xma = 1.1*max(max(X));             % Maximum X
Grid = linspace(xmi,xma,n);
f = zeros(100,100);
for i=1:length(alpha)
    for j=1:n
        for k=1:n
            f(j,k) = f(j,k) + alpha(i)*YC(i)*K([Grid(j) Grid(k)],XC(i,:));
        end
    end
end
hyp = f + b >= 0;
figure
mmin = abs(min(min(f+b))); mmax = max(max(f+b)); cmin = floor(50*mmin/(mmin+mmax)); cmax = ceil(50*mmax/(mmin+mmax));
cmap(1:cmin,:) = [linspace(0,1,cmin)' linspace(0,1,cmin)' ones(cmin,1)];
cmap(cmin+1:cmin+cmax,:) = [ones(cmax,1) linspace(1,0,cmax)' linspace(1,0,cmax)'];
image(Grid,Grid,f+b,'CDataMapping','scaled')
hold on
colormap(cmap)
contour(Grid,Grid,f+b,[0 0],'k-')
contour(Grid,Grid,f+b,[1 1],'r--')
contour(Grid,Grid,f+b,[-1 -1],'b--')
gscatter(X(:,2),X(:,1),Y,'br')
% gscatter(XC(:,2),XC(:,1),YC,'kk')
hold on
legend off
str = sprintf('Kernelized SVM');
title(str)
drawnow
figure
surf(f+b)
title('Decision surface')
end



