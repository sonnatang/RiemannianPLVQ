function distances = computeDistanceRieman(X, W)
%UNTITLED 
nb_samples = size(X,3);
nb_w = size(W,3);
distances = zeros(nb_samples,nb_w);
for i = 1:nb_samples
    for j = 1:nb_w
        distances(i,j) = Riemannian_dist(X(:,:,i),W(:,:,j));
    end
end

