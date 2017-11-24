function result=evalScriptWithPath(distanceFunctionName, evalPath)
    querymat=csvread([evalPath '/query/features.csv']);
    queryLab=csvread([evalPath '/query/labels.csv']);
    queryCam=csvread([evalPath '/query/cameras.csv']);

    testmat=csvread([evalPath '/test/features.csv']);
    testLab=csvread([evalPath '/test/labels.csv']);
    testCam=csvread([evalPath '/test/cameras.csv']);


    addpath(genpath('kReciprocalRerank'))
    addpath(genpath('marketEvaluation'))
    addpath(genpath('saquibRerank'))
    
    disp(evalPath)
    result=evalScript(distanceFunctionName, querymat, testmat, queryLab, testLab, queryCam, testCam);
    
 end