function [rdist, lqe_rdist]=diff_rank_dist(queryset, testset, k,lq,lq2)
 %[testset,queryset]= whiten_pls(1536); 
% load('./PersonReID/ResNet_Duke/WPCAmats.mat')
%   queryset=csvread('./PersonReID/inceptionV4/query/features.csv');
%   testset=csvread('./PersonReID/inceptionV4/test/features.csv');
% resnet 50-views-v2 on market 
% queryset=csvread('/cvhci/users/aeberle/results/tensorflow-models/market1501/active/2017-09-19_resnet_v1_50_views_v2/predictions-best/query/features.csv');
% testset=csvread('/cvhci/users/aeberle/results/tensorflow-models/market1501/active/2017-09-19_resnet_v1_50_views_v2/predictions-best/test/features.csv');
%  queryset=csvread('./PersonReID/ResNet_Duke/query/features.csv');
%  testset=csvread('./PersonReID/ResNet_Duke/test/features.csv');
%testset=l2_norm(testset); queryset=l2_norm(queryset);

nQuery=size(queryset,1); ntest=size(testset,1);
mat=[queryset ; testset];


dist=pdist2(mat,mat,'cosine'); 
[~, initial_rank]=sort(dist,2,'ascend'); 
clear mat; clear queryset; clear testset; clear dist
%orig_dist=dist(1:nQuery,nQuery+1:end);
r_dist=get_rank_dist(initial_rank,k);
%[~, initial_rank]=sort(r_dist,2,'ascend');

rdist=r_dist(1:nQuery,nQuery+1:end);   % rank dist query x test
% Local query expansion %%
top_lq_nb=initial_rank(:,2:lq+1);  % top 6 neighbour indxs
%lqe_rdist=zeros(nQuery,ntest); %zeros(size(rdist))

t_ind=top_lq_nb(nQuery+1:end,:).';  % test top lq nbr

next_2_tnbr=initial_rank(t_ind,2:lq2+1);   % lq2=lq/2   .. (lq*ntest,lq2)
next_2_tnbr=reshape(next_2_tnbr',[lq*lq2,ntest]);
t_ind=[t_ind;next_2_tnbr];

t_nbr_dist=r_dist(t_ind,1:nQuery); % dist of test top nbrs wrt to query: size is [(Lq x testsize) , nQuery]
t_nbr_dist=reshape(t_nbr_dist,[lq+lq*lq2,ntest,nQuery]);  %size. [:,:,nquery]

q_ind=top_lq_nb(1:nQuery,:).'; % query top lq nbr

next_2_qnbr=initial_rank(q_ind,2:lq2+1);   % lq2=lq/2   .. (lq*nQuery,lq2)
next_2_qnbr=reshape(next_2_qnbr',[lq*lq2,nQuery]);
q_ind=[q_ind;next_2_qnbr];

q_nbr_dist=r_dist(q_ind,nQuery+1:end); % dist of query top nbrs wrt to test: size is [(Lq x nQuery) , ntest]

q_nbr_dist=reshape(q_nbr_dist,[lq+lq*lq2,nQuery,ntest]);
q_nbr_dist=permute(q_nbr_dist,[1,3,2]);   %size. [:,:,nquery]

 lqe_rdist=squeeze(mean([q_nbr_dist;t_nbr_dist]));  %%%final distance: summing/average nbr distances for each query-test pair







% t_nbr_dist=r_dist(t_ind,nQuery+1:end); % (6*19732 x 19732)
% M = kron(eye(ntest),ones(lq,1));
%  t_dist=reshape(t_nbr_dist(logical(M)),[lq,ntest]);
%clear M;
%  for q=1:nQuery
% r_ind=top_lq_nb(q,:); 
% 
% lqe_rdist(q,:)=sum([ r_dist(r_ind,nQuery+1:end);  reshape(t_nbr_dist(:,q),[lq,ntest]);]); %rdist(q,:);
% %lqe_rdist(q,:)=mean([ r_dist(r_ind,nQuery+1:end); t_dist ]); %rdist(q,:);
% end

end