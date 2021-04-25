
function out = Riemannian_dist(p,x)
% Riemannian distance
% g = p^(-1)*x;
% [V,D] = eig(g);
% d = diag(D);
% out  = sum(log(d).^2);

[V,D] = eig(p,x);
d = diag(D);
out  = sum(log(d).^2);



% [u,Lambda] = eig(p);
% g = u*sqrt(Lambda);
% g_iv = g^-1;
% Y = g_iv*x*g_iv';
% [V,D] = eig(Y);
% d = diag(D);
% out2  = sum(log(d).^2);
end