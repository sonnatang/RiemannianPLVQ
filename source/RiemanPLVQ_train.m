function [ model,varargout] = RiemanPLVQ_train(trainSet,trainLab,varargin)
%train_RiemanProbLVQ.m - trains the Riemannian probablistic LVQ algorithm
%NOTE: minimal requirement version 7.4.0.336 (R2007a) 
%  example for usage:
%  trainSet is n times n times m array, containing m  n times n SPD matrix;
%  trainLab = [1;1;2];
%  model=train_RiemanProbLVQ(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = predict_Rieman(trainSet, model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  trainSet : matrix array with training samples in its 3rd dimension
%  trainLab : vector with the labels of the training set
% optional parameters:
%  PrototypesPerClass: (default=1) the number of prototypes per class used. This could
%  be a number or a vector with the number for each class
%  initialPrototypes : (default=[]) a set of prototypes to start with. If not given initialization near the class means
%  testSet           : (default=[]) an optional test set used to compute
%  the test error. The last column is expected to be a label vector
% parameter for the stochastic gradient descent sgd
%  nb_epochs             : (default=100) the number of epochs for sgd
%  learningRatePrototypes: (default=[]) the learning rate for the prototypes. 
%  Could be the start and end value used for a sigmoidal spectrum or a vector of length nb_epochs
%
% output: the model with prototypes w their labels c_w 
%  optional output:
%  initialization : a struct containing the settings
%  trainError     : error in the training set
%  testError      : error in the training set (only computed if 'testSet' is given)
%  costs          : the output of the cost function
% 
% Citation information:
% 
% Fengzhen Tang (adapted from the code written by Kerstin Bunte available at http://matlabserver.cs.rug.nl/gmlvqweb/web/)
%
% tangfengzhen@hotmail.com
% Sunday Apr 25 08:03 2021
%
% Conditions of GNU General Public License, version 2 and BSD License apply.
% See file 'license-gpl2.txt' and 'BSD_license.txt' enclosed in this package.
% Programs are not for use in critical applications!
%
nout = max(nargout,1)-1;
p = inputParser;   % Create an instance of the class.
p.addRequired('trainSet', @isfloat);
p.addRequired('trainLab', @(x) length(x)==size(trainSet,3) & isnumeric(x));
p.addOptional('testSet', [], @(x) (size(x,1)-1)==size(trainSet,1)&...
    (size(x,2)-1)==size(trainSet,2) & isfloat(x));
p.addParamValue('PrototypesPerClass', ones(1,length(unique(trainLab))), @(x)(sum(~(x-floor(x)))/length(x)==1 && (length(x)==length(unique(trainLab)) || length(x)==1)));
p.addParamValue('initialPrototypes',[], @(x)(size(x,2)==size(trainSet,2) && isfloat(x)));
p.addOptional('comparable', 0, @(x)(~(x-floor(x))));

% parameter for the stochastic gradient descent
p.addOptional('nb_epochs', 100, @(x)(~(x-floor(x))));
p.addParamValue('learningRatePrototypes', [], @(x)(isfloat(x) || isa(x,'function_handle')));%  && (length(x)==2 || length(x)==p.Results.epochs))
p.addParamValue('sigma2', [], @(x)(isfloat(x)));

p.FunctionName = 'RPLVQ';
% Parse and validate all input arguments.
p.parse(trainSet, trainLab, varargin{:});

%%% check if results should be comparable
if p.Results.comparable,
    rng('default');
end
% number of classes
classes = unique(trainLab);
nb_classes = length(classes);
[n,n1,nb_samples] = size(trainSet);%dimension of features
if size(trainLab,1)~=nb_samples, trainLab = trainLab';end
testSet = p.Results.testSet;

initialization = rmfield(p.Results, 'trainSet');
initialization.trainSet = [num2str(n), 'x', num2str(n1),...
    'x', num2str(nb_samples),' matrix'];
initialization = rmfield(initialization, 'trainLab');
initialization.trainLab = ['vector of length ',num2str(length(trainLab))];
if ~isempty(testSet)
    initialization = rmfield(initialization, 'testSet');
    initialization.testSet = [num2str(size(testSet,1)-1),'x',....
        num2str(size(testSet,2)-1),'x',num2str(size(testSet,3)),' matrix'];
end

% Display all arguments.
disp 'Settings for RPLVQ:'
disp(initialization);


PrototypePerClass =  p.Results.PrototypesPerClass;
nb_ppc = PrototypePerClass;% number of prototypes for each class 
if length(nb_ppc)~=nb_classes
    nb_ppc = ones(1,nb_classes)*nb_ppc;
end
nb_prot = sum(nb_ppc);

%%% initialize the prototypes
if isempty(p.Results.initialPrototypes)
    %intialize near the class centers
    w = zeros(n,n1,nb_prot);
    c_w = zeros(nb_prot,1);
    actPos = 1;
    for actClass = 1:nb_classes
        nb_prot_c = nb_ppc(actClass);
        classMean = riemann_mean(trainSet(:,:,trainLab==classes(actClass)));
        c_w(actPos:actPos+nb_prot_c-1) = classes(actClass);
        for i = 1:nb_prot_c       
            d = (2*rand(n,n) - ones(n,n))/1000;
            tp = triu(d);
            t = tp+tp'- diag(diag(tp));
            init_w= Exp(classMean,t);
            w(:,:,actPos) = init_w ;
            actPos = actPos+1;
        end

    end
else
    % initialize with given w
    w = p.Results.initialPrototypes(1:end-1,1:end-1,:);
    c_w = p.Results.initialPrototypes(end,end,:);
end
    

model = struct();
model.w = w;
model.c_w = c_w;
%model.nb_classes = nb_classes;
clear w c_w;

%%% gradient descent variables
nb_epochs = p.Results.nb_epochs;
% compute the vector of nb_epochs learning rates alpha for the prototype learning
if isa(p.Results.learningRatePrototypes,'function_handle')
    % with a given function specified from the user
    alphas = arrayfun(p.Results.learningRatePrototypes, 1:nb_epochs);
elseif length(p.Results.learningRatePrototypes)>2
    if length(p.Results.learningRatePrototypes)==nb_epochs
        alphas = p.Results.learningRatePrototypes;
    else
        disp('The learning rate vector for the prototypes does not fit the nb of epochs');
        return;
    end
else
    % or use an decay with a start and a decay value
    if isempty(p.Results.learningRatePrototypes)
       initialization.learningRatePrototypes = PrototypePerClass*[n/100, n/10000];
       % initialization.learningRatePrototypes = PrototypePerClass*[n/50, n/5000];
       % initialization.learningRatePrototypes = PrototypePerClass*[n/200, n/20000];
    end
    alpha_start = initialization.learningRatePrototypes(1);
    alpha_end = initialization.learningRatePrototypes(2);
    alphas = arrayfun(@(x) alpha_start * (alpha_end/alpha_start)^(x/nb_epochs), 1:nb_epochs);
%     alphas = arrayfun(@(x) alpha_start / (1+(x-1)*alpha_end), 1:nb_epochs);
end
%%% the variance sigma2
if isempty(p.Results.sigma2)
    dists = computeDistanceRieman(trainSet, model.w);
    sigma2_opt = 0.5*median(dists(:))+0.001*rand();    
    beta = 0.99;
    sigma2 = sigma2_opt + 0.2;
    sigma2s = zeros(nb_epochs,1);
    for t=1:nb_epochs
       if sigma2>=max(sigma2_opt - 0.2,0.01)
           beta = beta^1.1;
           sigma2 = sigma2*beta;
       end
       sigma2s(t) = sigma2;
    end
else
    sigma2s = p.Results.sigma2;
end  

if length(sigma2s)~=nb_epochs
    sigma2s = ones(nb_epochs,1)*sigma2s;
end
sigma2 = sigma2s(1);
model.sigma2 = sigma2;
%%% initialize requested outputs
costs = [];
trainError = [];
testError = [];

if nout>=2
    % cost requested
   costs(1) = RiemanPLVQ_costfun(trainSet,trainLab,model);
   if nout>=3
       % train error requested
        trainError = ones(1,nb_epochs+1);
        estimatedLabels = RiemanPLVQ_classify(trainSet, model); % error after initialization
        trainError(1) = sum( trainLab ~= estimatedLabels )/nb_samples;
        if nout>=4
            % test error requested
            if isempty(testSet)
                testError = [];
                disp('The test error is requested, but no labeled test set given. Omitting the computation.');
            else
                testError = ones(1,nb_epochs+1);
                estimatedLabels = RiemanPLVQ_classify(testSet(1:end-1,1:end-1,:), model); % error after initialization
                testError(1) = sum(squeeze(testSet(end,end,:))~= estimatedLabels )/length(estimatedLabels);
            end        
        end
   end
end


%%%optimize with stochastic gradient descent
for epoch=1:nb_epochs
    %time = cputime;
    if mod(epoch,100)==0, disp(epoch); end
    %generate order to sweep through the training set
    order = randperm(nb_samples);
    sigma2 = sigma2s(epoch);
    model.sigma2 = sigma2;
    beta = 0.5/sigma2;
    for  i=1:nb_samples
        xi = trainSet(:,:,order(i));
        c_xi = trainLab(order(i));
        %time = cputime;
        dist = computeDistanceRieman(xi,model.w);
        %%%compute class probability
%         fs = -beta*dist;
%         fsmax = max(fs);
%         prob = exp(fs-fsmax);
%         probx = prob/sum(prob);
%         fsy = fs;
%         fsy(~(model.c_w == c_xi)) = [];
%         fsymax = max(fsy);
%         probyy = exp(fsy - fsymax)
%         probyy = probyy/sum(probyy);
%         proby = zeros(size(prob));
%	      proby(model.c_w == c_xi) = probyy; 
   
        prob = exp(-beta*dist); %changes by Fengzhen 2020-1-15 10:28   
        proby = prob;
        proby(~(model.c_w == c_xi))=0;
        proby = proby/sum(proby);       
        probx = prob/sum(prob);
        alpha = alphas(epoch);
        if isnan(sum(proby)) | isnan(sum(probx))
            warning('nan probability, no update here');
            continue;  % no updates
        end
        %% update prototypes:
        for J = 1:nb_prot
            wJ = model.w(:,:,J);
            c_wJ = model.c_w(J);
            if c_wJ == c_xi
              probt = (proby(J)-probx(J));
            else
                probt = - probx(J);
            end
             % Tangent space mapping
             XJ = 2*alpha*beta*probt*Log(wJ,xi);
             try
                 W = Exp(wJ,XJ);
             catch
                  warning('bad tangent vector, no update here');
                  W = model.w(:,:,J);
             end
             model.w(:,:,J) = W;
         end
        
        
    end
     %trtime = cputime-time
    
    if nout>=2
        % cost requested
       costs(epoch+1) = RiemanPLVQ_costfun(trainSet,trainLab,model);
       if nout>=3
           % train error requested
            estimatedLabels = RiemanPLVQ_classify(trainSet, model); % error after initialization
            trainError(epoch+1) = sum( trainLab ~= estimatedLabels )/nb_samples;
            if nout>=4
                % test error requested
                if isempty(testSet)
                    testError = [];
                    disp('The test error is requested, but no labeled test set given. Omitting the computation.');
                else
                    estimatedLabels = RiemanPLVQ_classify(testSet(1:end-1,1:end-1,:), model); % error after initialization
                    testError(epoch+1) = sum(squeeze(testSet(end,end,:))~= estimatedLabels )/length(estimatedLabels);
                end        
            end
       end
    end


   
    
end

%%output of the training
varargout = cell(nout,1);
for k=1:nout
	switch(k)
		case(1)
			varargout(k) = {initialization};
		case(2)
            varargout(k) = {costs};
        case(3)
            varargout(k) = {trainError};
        case(4)
            varargout(k) = {testError};
	end
end



