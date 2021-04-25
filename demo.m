%%% demo of Riemannian GLVQ
addpath('./source')
datadir = './data/BCIIV2a/';

dir = './res/';
%%run PLRSQ
fname = 'CV_normF10_30CA01';
load([datadir fname '.mat']);

trainIdx = ~testIdx;

trainP = P(:,:,trainIdx);
trainLab = Label(trainIdx);

testP = P(:,:,testIdx);
testLab = Label(testIdx);

nPrototype = 1;%needs to specify
nb_epochs = 100;%needs to specify
sigma2 = 9.5% needs to specify
%
classes = unique(trainLab);
testSetLab = zeros(size(testP,1)+1,size(testP,2)+1,size(testP,3));
testSetLab(1:end-1,1:end-1,:) = testP;
testSetLab(end,end,:) = testLab;

 [ model,setting,costs,trainError,testError] = ...
           RiemanPLVQ_train(trainP,trainLab,'PrototypesPerClass',nPrototype,...
               'nb_epochs',nb_epochs,'testSet',testSetLab,'sigma2',sigma2);
           

%%%training 
predtrainLab  = RiemanPLVQ_classify(trainP,model);
trainacc = evaluation_measures(trainLab,predtrainLab,classes, 'RA' );
trainkappa= evaluation_measures(trainLab,predtrainLab,classes, 'KAPPA' );
fprintf('RPLVQ: accuracy on the training set: %f\n',trainacc);
fprintf('RPLVQ: kappa on the training set: %f\n',trainkappa);
%%%test
[predLab] = RiemanPLVQ_classify(testP, model);
testacc = evaluation_measures(testLab, predLab,classes, 'RA' );
testkappa = evaluation_measures(testLab, predLab,classes, 'KAPPA' );
fprintf('RPLVQ: accuracy on the test set: %f\n',testacc);
fprintf('RPLVQ: kappa on the test set: %f\n',testkappa);

save([dir fname   'ResRPLVQ_P' num2str(nPrototype) 'Iter' num2str(nb_epochs) ...
     'sigma2_' num2str(sigma2)  '.mat'],...
     'testacc','trainacc','trainkappa','testkappa','costs','trainError','testError');
