
function Rank_sim=get_rank_dist(initial_rank,k)



Rank_sim=compute_RL_score(initial_rank, initial_rank,k);
%convert the similarites to distance
 Rank_sim=min_max_norm(Rank_sim);
 Rank_sim=(1-Rank_sim) ;
 disp('prepared dist mats and rank lists.. commencing to match') 



function R=compute_RL_score(L1,L2,k)
[~,pos_L1]=sort(L1,2,'ascend'); [~,pos_L2]=sort(L2,2,'ascend');  %matrix
fac_1=(max(0,(k+1- pos_L1)));
fac_2=(max(0, (k+1- pos_L2)));
 
clear pos_L1; clear pos_L2;
R=fac_1*fac_2';
clear fac_1, clear fac_2;
end

end