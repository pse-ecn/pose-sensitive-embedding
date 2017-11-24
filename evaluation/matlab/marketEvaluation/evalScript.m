function result=evalScript(distanceFunctionName, querymat, testmat, queryID, testID, queryCAM, testCAM)

distFunction = str2func(distanceFunctionName);
dist = distFunction(querymat, testmat);

nQuery=size(querymat,1); nTest=size(testmat,1);
result = evaluateWithDistances(nQuery, nTest, queryID, testID, queryCAM, testCAM, dist);