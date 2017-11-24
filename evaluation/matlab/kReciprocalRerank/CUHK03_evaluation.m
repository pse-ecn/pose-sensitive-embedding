clc;clear all;close all;
%**************************************************%
% This code implements IDE baseline for the CUHK03 %
% dataset under the new traning/testing protocol.  %
% Please modify the path to your own folder.       %
% We use the mAP and rank-1 rate as evaluation     %
%**************************************************%
% if you find this code useful in your research, please kindly cite our
% paper as,
% Zhun Zhong, Liang Zheng, Donglin Cao, Shaozi Li,
% Re-ranking Person Re-identification with k-reciprocal Encoding, CVPR, 2017.


% Please download CUHK03 dataset and unzip it in the "dataset/CUHK03" folder.
addpath(genpath('LOMO_XQDA/'));
run('KISSME/toolbox/init.m');
addpath(genpath('utils/'));

%% re-ranking setting
k1 = 20;
k2 = 6;
lambda = 0.3;

%% network name
evalPath = 'D:\development\private\masters\results\kReciprocal\CUHK03';

netname = 'ResNet_50'; % network: CaffeNet  or ResNet_50 googlenet

detected_or_labeled = 'detected'; % detected/labeled
load(['data/CUHK03/cuhk03_new_protocol_config_' detected_or_labeled '.mat']);


%% load feature Deep feature
% feat = importdata([evalPath '/' netname '_IDE_' detected_or_labeled '.mat']);
% feat = double(feat);

%% load feature LOMO feature
feat = importdata([evalPath '/cuhk03_' detected_or_labeled '_lomo.mat']);
feat = single(feat.descriptors);

%% train info
label_train = labels(train_idx);
cam_train = camId(train_idx);
train_feature = feat(:, train_idx);
%% test info
galFea = feat(:, gallery_idx);
probFea = feat(:, query_idx);
label_gallery = labels(gallery_idx);
label_query = labels(query_idx);
cam_gallery = camId(gallery_idx);
cam_query = camId(query_idx);

%% normalize
sum_val = sqrt(sum(galFea.^2));
for n = 1:size(galFea, 1)
    galFea(n, :) = galFea(n, :)./sum_val;
end

sum_val = sqrt(sum(probFea.^2));
for n = 1:size(probFea, 1)
    probFea(n, :) = probFea(n, :)./sum_val;
end

sum_val = sqrt(sum(train_feature.^2));
for n = 1:size(train_feature, 1)
    train_feature(n, :) = train_feature(n, :)./sum_val;
end


%% Euclidean
%dist_eu = pdist2(galFea', probFea');
my_pdist2 = @(A, B) sqrt( bsxfun(@plus, sum(A.^2, 2), sum(B.^2, 2)') - 2*(A*B'));
dist_eu = my_pdist2(galFea', probFea');
evaluate(dist_eu, ['The IDE (' netname ') + Euclidean performance:\n'], label_gallery, label_query, cam_gallery, cam_query)


%% Euclidean + re-ranking
query_num = size(probFea, 2);
dist_eu_re = re_ranking( [probFea galFea], 1, 1, query_num, k1, k2, lambda);
evaluate(dist_eu_re, ['The IDE (' netname ') + Euclidean + re-ranking performance:\n'], label_gallery, label_query, cam_gallery, cam_query)


addpath(genpath('../kReciprocalRerank'))
addpath(genpath('../marketEvaluation'))
addpath(genpath('../saquibRerank'))

    
querymat = probFea';
testmat = galFea';

[~,lqe_dist] = direct_LQE_dist_comm(querymat, testmat, 3, 8);

[rdist,lqe_rdist] = diff_rank_dist(querymat, testmat, 25,3,8);

evaluate(rdist', 'rdist\n', label_gallery, label_query, cam_gallery, cam_query)
evaluate(lqe_dist, 'LQE\n', label_gallery, label_query, cam_gallery, cam_query)
evaluate(lqe_rdist, 'rankLQE\n', label_gallery, label_query, cam_gallery, cam_query)


function evaluate(dist, name, label_gallery, label_query, cam_gallery, cam_query)
    [CMC_eu_re, map_eu_re, ~, ~] = evaluation(dist, label_gallery, label_query, cam_gallery, cam_query);

    fprintf(name);
    fprintf('  mAP,  Rank1,  Rank5,  Rank10,  Rank50\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', map_eu_re(1)*100, CMC_eu_re(1) * 100, CMC_eu_re(5) * 100, CMC_eu_re(10) * 100, CMC_eu_re(50) * 100);
end

