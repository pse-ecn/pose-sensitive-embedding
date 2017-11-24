function datamat=l2_norm(datamat)
        ng=sqrt(sum(datamat.^2,2));
        datamat=bsxfun(@rdivide,datamat,ng);
end