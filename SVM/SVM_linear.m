clear 
close all
clc
str = 'svm_data.mat';
load(str)
X = (X-mean(X))./sqrt(var(X));
%% Plot data
n = length(X(:,1));
%% Determine Alpha's (using Digression Coordinate Ascent)
alpha = zeros(n,1);
gamma = 1;
C = 1;
tol = 1e-6;
z = 0;
w = [0;0]; b = 0; bias = 0;
while (1) % as long as KKT aren't fulfilled keep running
    changed_alphas = 0;
    n = size(X(:,1));
    for i=1:n
    %% Check error
        Ei = sum(alpha.*Y.*X,1)*X(i,:)' + b;
        %% KKT Conditions
        if (alpha(i) == 0 && Ei*Y(i) < 1) || (alpha(i)<=C && Ei*Y(i) ~= 1) || (alpha(i) >= C-tol && Ei*Y(i) > 1) %% (Y(i)*Ei < -tol && alpha(i) < C) || (Y(i)*Ei > tol && alpha(i) > 0)%%
            % if not fulfilled, run for entire training group except i
            for j=[1:i-1,i+1:n]
                % Error of match
                Ei = sum(alpha.*Y.*X,1)*X(i,:)'+(max(w'*X') - min(w'*X'))/2 - Y(i);
            Ej = sum(alpha.*Y.*X,1)*X(j,:)'+(max(w'*X') - min(w'*X'))/2 - Y(j);
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
            eta = 2*X(j,:)*X(i,:)' - X(j,:)*X(j,:)' - X(i,:)*X(i,:)';
            % if positive we don't improve
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
            alpha(i)=alpha(i) + gamma*Y(i)*Y(j)*(a_jhold-alpha(j));
            w = sum(alpha.*Y.*X)';
            b = (max(w'*X') + min(w'*X'))/2;
            changed_alphas=changed_alphas+1;
            end
        end            
    end
    z = z+1;
    fprintf('Iteration %.0f done, %.0f alphas changed \n',z,changed_alphas)
    if changed_alphas==0
        break
    end
    X=X((find(alpha~=0)),:);
    Y=Y((find(alpha~=0)),:);
    alpha=alpha((find(alpha~=0)),:);
end
%% Calculate Hyperplane
XC = X; YC = Y;
load(str)
X = (X-mean(X))./sqrt(var(X));
K = @(X,Y) X*Y';
visualize(XC,YC,alpha,b,K,X,Y)
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
hyp = f - b >= 0;
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
hold on
legend off
str = sprintf('Linear SVM');
title(str)
drawnow
figure
surf(f+b)
title('Decision surface')
end

