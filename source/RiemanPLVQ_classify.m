function [predictLabel] = RiemanPLVQ_classify(testSet, model)
%%RiemanPLVQ_classify.m - classifies the given data with the given model
%  example for usage:
%  trainSet is n times n times m array, containing m  n times n SPD matrix
%  trainLab = [1;1;2;...];
%  model=RiemanPLVQ_train(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = RiemanPLVQ_classify(trainSet, model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  testSet :  matrix array with training samples in its 3rd dimension
%  model    : RiemanPLVQ model with prototypes w their labels c_w 
% 
% output    : the estimated labels
%  
% Fengzhen Tang (adapted from the code written by Kerstin Bunte available ...
% at http://matlabserver.cs.rug.nl/gmlvqweb/web/)
% tangfengzhen@sia.cn
% Monday Dec 7 08:27 2020
%
%
d = computeDistanceRieman(testSet,model.w);
[min_v,min_id] = min(d,[],2);
predictLabel = model.c_w(min_id);

