function result=evaluateWithDistances(nQuery, nTest, queryID, testID, queryCAM, testCAM, dist)
%% search the database and calcuate re-id accuracy

CMC = zeros(nQuery, nTest);
ap = zeros(nQuery, 1); % average precision

for k = 1:nQuery
    % load groud truth for each query (good and junk)
    good_index = intersect(find(testID == queryID(k)), find(testCAM ~= queryCAM(k)))';% images with the same ID but different camera from the query
    junk_index1 = find(testID == -1);% images neither good nor bad in terms of bbox quality
    junk_index2 = intersect(find(testID == queryID(k)), find(testCAM == queryCAM(k))); % images with the same ID and the same camera as the query
    junk_index = [junk_index1; junk_index2]';
    tic
    score = dist(:, k);
    
    
    % sort database images according Euclidean distance
    [~, index] = sort(score, 'ascend');  % single query
    
    
    [ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);% compute AP for single query
    
    %ap_pairwise(k, :) = compute_AP_multiCam(good_index, junk_index, index, queryCam(k), testCam); % compute pairwise AP for single query
       

    %%%%%%%%%%%%%% calculate r1 precision %%%%%%%%%%%%%%%%%%%%
end

CMC = mean(CMC);

result.mAP = 100 * mean(ap);
result.rec_rates = 100 * CMC;

fprintf('mAP = %2.2f, Rank-1 = %2.2f, Rank-5 = %2.2f , Rank-10 = %2.2f, Rank-50= %2.2f\n', result.mAP, result.rec_rates(1), result.rec_rates(5), result.rec_rates(10), result.rec_rates(50));