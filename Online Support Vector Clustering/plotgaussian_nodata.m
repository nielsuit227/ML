function [] = plotgaussian_nodata(Mu, Cov)
[n,m] = size(Mu);
Mu = reshape(Mu,max(n,m),1);
xmin = -10;
xmax = 10;
n = 100;
grid = linspace(xmin, xmax, n);
f = zeros(n,n);
for i =1:n
    for j = 1:n
        f(i,j) = 1/sqrt(det(2*pi*Cov))*exp(-1/2*([grid(i);grid(j)]-Mu)'/Cov*([grid(i);grid(j)]-Mu));
    end
end
surf(f)
end