%%% 21st of March, Niels Uitterdijk. Script that builds KD tree. Will be
%%% extended to Randomized KD tree and Priority Search K-Means Tree in two
%%% other files. Data is under NDA and owned by Tritium Pty Ltd. The point
%%% of KD tree's is to get priority queue's and help with finding K-Nearest
%%% Neighbors or Radius-Nearest Neighbours. Dimension is reduced to 2D for
%%% visualization. 
%% Initialize
clear
close all
clc
load data
X(:,1:2) = [];
X = (X-mean(X))./sqrt(var(X));
X = X(1:10000,:);
%% PCA
[~,S,V] = svd(X);
X = X*V(:,1:2);
clear S V
figure
gscatter(X(:,1),X(:,2))
drawnow
%% Tree
Xvar = var(X);
[~,KDind] = sort(Xvar);
[n,m] = size(X);

Tree = [];
[P,Tree] = build_tree(Tree, X, 0)

function [P, Tree] = build_tree(Tree, points, depth)
[n,m] = size(points);
axis = rem(depth,m)+1;
[~,ind] = sort(points(:,axis));
sorted = points(ind,:);
Tree = [Tree; sorted(round(n/2),:)];
P(ind(1:round(n/2)-1)) = 

end
