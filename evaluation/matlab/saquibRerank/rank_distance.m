function rank_dist = rank_distance(queryset, testset)

 [rank_dist, ~]=diff_rank_dist(queryset, testset, 25,3,8);