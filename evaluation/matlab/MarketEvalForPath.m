function result=evalMarketWithPath(evalPath)
    querymat=csvread([evalPath '/query/features.csv']);
    queryLab=csvread([evalPath '/query/labels.csv']);
    queryCam=csvread([evalPath '/query/cameras.csv']);

    testmat=csvread([evalPath '/test/features.csv']);
    testLab=csvread([evalPath '/test/labels.csv']);
    testCam=csvread([evalPath '/test/cameras.csv']);

    
    disp(evalPath)
    
    noRerankingDist = pdist2(testmat, querymat, 'cosine');
    [result.rec_rates, result.mAP, ~, ~] = evaluation(noRerankingDist, testLab, queryLab, testCam, queryCam);
 end