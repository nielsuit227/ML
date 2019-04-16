function [] = plotgaussian(X,Mu,Cov)
[n,m] = size(X);
X0 = X - Mu;
[V D] = eig(Cov);
[D order] = sort(diag(D),'descend');
D = diag(D);
V = V(:,order);
t = linspace(0,2*pi,100);
e = [cos(t);sin(t)];
VV = 2*V*sqrt(D);
e = VV*e+Mu';
i = rand;
plot(e(1,:),e(2,:),'k')
hold on
gscatter(X(:,1),X(:,2),ones(n,1));
axis equal
end