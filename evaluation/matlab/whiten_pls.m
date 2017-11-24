function [result]= whiten_pls(distanceFunctionName, basePath, nc)

path(pathdef); 
addpath(genpath('marketEvaluation'))

%% dataloader
queryset=csvread([basePath 'query/features.csv']);
testset=csvread([basePath 'test/features.csv']);

trainset= csvread([basePath 'train/features.csv']);
trainLab=csvread([basePath 'train/labels.csv']);

queryLab=csvread([basePath 'query/labels.csv']);
testLab=csvread([basePath 'test/labels.csv']);

queryCam=csvread([basePath 'query/cameras.csv']);
testCam=csvread([basePath 'test/cameras.csv']);

%%%
 warning off
% % % Training whiten PCA and PLS %%
%   featLength=size(trainset,2);
% % %% apply pca %%
 %nc=1000;%ceil(0.2*size(trainset,2));
 [pcaMat,~,ig_] = pca(trainset,'Centered',false,'NumComponents',nc);
 whiten_=inv(sqrt(diag(ig_)));  whiten_=whiten_(1:nc,1:nc);
 
 trainset=(trainset*pcaMat)*whiten_;
 testset=(testset*pcaMat)*whiten_;
 queryset=(queryset*pcaMat)*whiten_;

%save('./PersonReID/Marktmats','trainset','testset', 'queryset')

%trainset=zscore(trainset); testset=zscore(testset); queryset=zscore(queryset);
% disp('going to train the pls model now ...')
% 
% options = statset('UseParallel','always','UseSubstreams','always','Streams',{RandStream('mlfg6331_64')});
% [XL,YL,XS,YS,~,~,~,plsG] = plsregress(trainset,trainLab,60,'options',options);
% Bv=plsG.W*(inv(XL'*plsG.W));
% trainset=trainset*Bv; testset=testset*Bv;  queryset=queryset*Bv; 

%trainset=zscore(trainset); 

testset=zscore(testset); queryset=zscore(queryset);
%fprintf('features are projected from %d to %d-dimensional using whitenend PCA\n', featLength,nc);


addpath(genpath('kReciprocalRerank'))
addpath(genpath('saquibRerank'))

result=evalScript(distanceFunctionName, queryset, testset, queryLab, testLab, queryCam, testCam);

end


