%% This code was inspired by the code of
%% Zhun Zhong, Liang Zheng, Donglin Cao, Shaozi Li,
%% Re-ranking Person Re-identification with k-reciprocal Encoding, CVPR, 2017.

function result = CUHK03EvalForPath(evalPath)
    addpath(genpath('utils'))

    detected_or_labeled = 'labeled'; % detected/labeled
    protocol = load(['kReciprocalRerank/data/CUHK03/cuhk03_new_protocol_config_' detected_or_labeled '.mat']);

    queryMat=csvread([evalPath '/query/features.csv']);
    queryNames = readFileByLines([evalPath '/query/names.csv']);
    expectedQueryNames = string(protocol.filelist(protocol.query_idx));
    [~, idx] = ismember(expectedQueryNames, queryNames);
    queryMatOrdered = queryMat(idx, :);

    testMat=csvread([evalPath '/test/features.csv']);
    testNames = readFileByLines([evalPath '/test/names.csv']);
    expectedTestNames = string(protocol.filelist(protocol.gallery_idx));
    [~, idx] = ismember(expectedTestNames, testNames);
    testMatOrdered = testMat(idx, :);



    %% test info
    galFea = testMatOrdered'; % features x gal_size
    probFea = queryMatOrdered'; % features x query_size
    label_gallery = protocol.labels(protocol.gallery_idx);
    label_query = protocol.labels(protocol.query_idx);
    cam_gallery = protocol.camId(protocol.gallery_idx);
    cam_query = protocol.camId(protocol.query_idx);

    %% normalize
    sum_val = sqrt(sum(galFea.^2));
    for n = 1:size(galFea, 1)
        galFea(n, :) = galFea(n, :)./sum_val;
    end

    sum_val = sqrt(sum(probFea.^2));
    for n = 1:size(probFea, 1)
        probFea(n, :) = probFea(n, :)./sum_val;
    end


    %% Euclidean
    %dist_eu = pdist2(galFea', probFea');
    my_pdist2 = @(A, B) sqrt( bsxfun(@plus, sum(A.^2, 2), sum(B.^2, 2)') - 2*(A*B'));
    dist_eu = my_pdist2(galFea', probFea');
    result = evaluate(dist_eu, 'The Euclidean performance:\n', label_gallery, label_query, cam_gallery, cam_query);
    
    
    
    function result = evaluate(dist, name, label_gallery, label_query, cam_gallery, cam_query)
        [CMC_eu_re, map_eu_re, ~, ~] = evaluation(dist, label_gallery, label_query, cam_gallery, cam_query);

        fprintf(name);
        fprintf('  mAP,  Rank1,  Rank5,  Rank10,  Rank50\n');
        fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', map_eu_re(1)*100, CMC_eu_re(1) * 100, CMC_eu_re(5) * 100, CMC_eu_re(10) * 100, CMC_eu_re(50) * 100);
        
        result.mAP = map_eu_re(1)*100;
        result.rec_rates = 100 * CMC_eu_re;
    end


end
