function dist=kReciprocalDistance(querymat, testmat)

% addpath(genpath('utils/'));
% addpath(genpath('LOMO_XQDA/code'));
% 

probFea=querymat'; galFea=testmat';
% k-reciprocal re-ranking
%% re-ranking setting
k1 = 20;
k2 = 6;
lambda = 0.3;
query_num = size(probFea, 2);


sum_val = sqrt(sum(galFea.^2));
for n = 1:size(galFea, 1)
    galFea(n, :) = galFea(n, :)./sum_val;
end

sum_val = sqrt(sum(probFea.^2));
for n = 1:size(probFea, 1)
    probFea(n, :) = probFea(n, :)./sum_val;
end


dist = re_ranking( [probFea galFea], 1, 1, query_num, k1, k2, lambda);
