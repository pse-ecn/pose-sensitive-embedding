function dist = marketDistance(querymat, testmat)

testmat=l2_norm(testmat).'; querymat=l2_norm(querymat).';
dist = sqdist(testmat, querymat); % distance calculate with single query. Note that Euclidean distance is equivalent to cosine distance if vectors are l2-normalized
