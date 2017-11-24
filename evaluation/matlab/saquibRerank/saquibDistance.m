function dist = saquibDistance(querymat, testmat)

addpath(genpath('saquibRerank/'));
[Rank_sim, dist,orig_dist]=checkRanksim(querymat,testmat, 26, 0.1);

