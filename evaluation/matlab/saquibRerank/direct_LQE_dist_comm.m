function [orig_dist,lqe_dist]=direct_LQE_dist_comm(queryset, testset,lq,lq2)
%  queryset=csvread('./PersonReID/inceptionV4/query/features.csv');
%   testset=csvread('./PersonReID/inceptionV4/test/features.csv');
% queryset=csvread('./PersonReID/ResNet_Duke/query/features.csv');
%  testset=csvread('./PersonReID/ResNet_Duke/test/features.csv');
 
%  queryset=csvread('/cvhci/users/aeberle/results/tensorflow-models/market1501/active/2017-09-19_resnet_v1_50_views_v2/predictions-best/query/features.csv');
% testset=csvread('/cvhci/users/aeberle/results/tensorflow-models/market1501/active/2017-09-19_resnet_v1_50_views_v2/predictions-best/test/features.csv');
 %testset=l2_norm(testset); queryset=l2_norm(queryset);
%oad('./PersonReID/inceptionV4/WPCAmats.mat')

nQuery=size(queryset,1); ntest=size(testset,1);
mat=[queryset ; testset];

  %dist = MahDist(1, mat,mat);
%  [~, initial_rank] = sort(dist, 2, 'ascend');

dist=pdist2(mat,mat,'cosine'); 
[~, initial_rank]=sort(dist,2,'ascend'); 
clear mat; clear queryset; clear testset;

%r_dist=get_rank_dist(initial_rank,k);
%[~, initial_rank]=sort(r_dist,2,'ascend');

orig_dist=dist(1:nQuery,nQuery+1:end);   % rank dist query x test
% Local query expansion %%
top_lq_nb=initial_rank(:,2:lq+1);  % top 6 neighbour indxs
%lqe_dist=zeros(nQuery,ntest); %zeros(size(rdist))

t_ind=top_lq_nb(nQuery+1:end,:).';  % test top lq nbr

next_2_tnbr=initial_rank(t_ind,2:lq2+1);   % lq2=lq/2   .. (lq*ntest,lq2)
next_2_tnbr=reshape(next_2_tnbr',[lq*lq2,ntest]);
t_ind=[t_ind;next_2_tnbr];

t_nbr_dist=dist(t_ind,1:nQuery); % dist of test top nbrs wrt to query: size is [(Lq x ntest) , nQuery]
t_nbr_dist=reshape(t_nbr_dist,[lq+lq*lq2,ntest,nQuery]);  %size. [:,:,nquery]

q_ind=top_lq_nb(1:nQuery,:).'; % query top lq nbr

next_2_qnbr=initial_rank(q_ind,2:lq2+1);   % lq2=lq/2   .. (lq*nQuery,lq2)
next_2_qnbr=reshape(next_2_qnbr',[lq*lq2,nQuery]);
q_ind=[q_ind;next_2_qnbr];

q_nbr_dist=dist(q_ind,nQuery+1:end); % dist of query top nbrs wrt to test: size is [(Lq x nQuery) , ntest]

q_nbr_dist=reshape(q_nbr_dist,[lq+lq*lq2,nQuery,ntest]);
q_nbr_dist=permute(q_nbr_dist,[1,3,2]);   %size. [:,:,nquery]

 lqe_dist=squeeze(mean([q_nbr_dist;t_nbr_dist]));  %%%final distance: by summing or average nbr distances for each query-test pair

% h=1;
%  for q=1:nQuery
% 
% lqe_dist(q,:)=sum([  q_nbr_dist(h:h+(lq-1),:) ;  reshape(t_nbr_dist(:,q),[lq,ntest]) ]); %
% h=lq+h;
% end


end