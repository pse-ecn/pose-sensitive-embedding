function [mat]=min_max_norm(mat)

   mat= bsxfun(@minus,mat,min(mat,[],2)); %subtract each row min from each row
    max_mat=max(mat,[],2); min_mat=min(mat,[],2); % geting max and min 
    max_min=max_mat-min_mat;
    
   mat=bsxfun(@rdivide,mat,max_min);
end