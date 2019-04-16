%# generate data
clear
close all
clc

num = 50000;
X = [ mvnrnd([0 1.5], [0.5 0.03 ; 0.03 0.025], num) ; ...
      mvnrnd([1 0.5], [0.09 -0.01 ; -0.01 0.09], num)   ];
G = [1*ones(num,1) ; 2*ones(num,1)];
Gt = G + round(rand(num*2,1));
gscatter(X(:,1), X(:,2), G,'br','oo')
axis equal, hold on

for k=1:2
    %# indices of points in this group
    idx = ( G == k );

    %# substract mean
    Mu = mean( X(idx,:) );
    X0 = bsxfun(@minus, X(idx,:), Mu);

    %# eigen decomposition [sorted by eigen values]
    [V D] = eig( X0'*X0 ./ (sum(idx)-1) );     %#' cov(X0)
    [D order] = sort(diag(D), 'descend');
    D = diag(D);
    V = V(:, order);

    t = linspace(0,2*pi,100);
    e = [cos(t) ; sin(t)];        %# unit circle
    VV = 2*V*sqrt(D);               %# scale eigenvectors
    e = bsxfun(@plus, VV*e, Mu'); %#' project circle back to orig space

    %# plot cov and major/minor axes
    i = rand;
    plot(e(1,:), e(2,:), 'Color','k');
    %#quiver(Mu(1),Mu(2), VV(1,1),VV(2,1), 'Color','k')
    %#quiver(Mu(1),Mu(2), VV(1,2),VV(2,2), 'Color','k')
end
Y = G; Y(Y==2) = -1;
save('Data.mat','X','Y')
