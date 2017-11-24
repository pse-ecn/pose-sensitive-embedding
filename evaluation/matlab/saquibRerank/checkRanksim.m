function  [Rank_sim,rank_dist,orig_dist]=checkRanksim(queryset,testset,k,lam)

nQuery=size(queryset,1);
mat=[queryset;testset];

% dist = MahDist(1, mat, mat);
% [~, initial_rank] = sort(dist, 2, 'ascend');

dist=pdist2(mat,mat,'cosine'); 
[~, initial_rank]=sort(dist,2,'ascend');
orig_dist=dist(1:nQuery,nQuery+1:end);
%  q_dist=pdist2(queryset,testset,'cosine'); %q_dist(logical(eye(size(q_dist))))=[]; q_dist=reshape(q_dist,19731,3368).';
%  t_dist=pdist2(testset,testset,'cosine');  %t_dist(logical(eye(size(t_dist))))=[];t_dist=reshape(t_dist,19731,19732).';
% % 
%  [~, L_q]=sort(q_dist,2,'ascend'); %L_q rank list
%  [~, L_t]=sort(t_dist,2,'ascend');
disp('prepared dist mats and rank lists.. commencing to match')
L_q=initial_rank(1:nQuery,:); L_t=initial_rank(nQuery+1:end,:); 
%Rank_sim=zeros(size(L_q,1),size(L_t,1));

Rank_sim=compute_RL_score(L_q, L_t,k);


   Rank_sim=min_max_norm(Rank_sim);
   Rank_sim=(1-Rank_sim)' ;
   rank_dist=(orig_dist'*lam)+(Rank_sim*(1-lam));



function R=compute_RL_score(L1,L2,k)
%N(t)=[]; R=0;

[~,pos_L1]=sort(L1,2,'ascend'); [~,pos_L2]=sort(L2,2,'ascend');  %matrix
fac_1=(max(0,(k+1- pos_L1)))./k;
 fac_2=(max(0, (k+1- pos_L2)))./k;
 
R=fac_1*fac_2';
end

end

