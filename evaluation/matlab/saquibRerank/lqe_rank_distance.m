function lqe_rdist = lqe_rank_distance(queryset, testset)

 [~, lqe_rdist]=diff_rank_dist(queryset, testset, 25,3,8);