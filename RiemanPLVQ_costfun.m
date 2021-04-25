function cost = RiemanPLVQ_costfun(trainSet,trainLab,model)
%%RiemanPLVQ_costfun.m - computes the costs for a given training set and
% RPLVQ model
%
% input: 
%  testSet :  matrix array with training samples in its 3rd dimension
%  model    : RiemanPLVQ model with prototypes w their labels c_w 
%  sigma2: the teperature parameter
%  output    : the estimated labels
%  
% @Fengzhen Tang
% tangfengzhen@sia.cn
% Monday Dec 7 08:27 2020
nb_samples = length(trainLab);
% labels should be a row vector
if size(trainLab,1)~=nb_samples, trainLab = trainLab';end
LabelEqPrototype = bsxfun(@eq,trainLab,model.c_w');
dists = computeDistanceRieman(trainSet, model.w);
prob = exp(-0.5*dists/model.sigma2);
Dcorrect = prob;
Dcorrect(~LabelEqPrototype) = 0;
probture = sum(Dcorrect,2);
norm = sum(prob,2);
logR = log(probture) - log(norm);
cost = -sum(logR);

end


