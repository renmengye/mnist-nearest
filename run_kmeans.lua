local mnist = require('mnist')
local utils = require('utils')
local kmeans = require('kmeans')

local train, test = mnist.loadData()
local trainDataResize = mnist.flattenFloat(train.data)
local testDataResize = mnist.flattenFloat(test.data)
local K = 20
local numClasses = 10
local numIter= 20

local means = kmeans.run(K, trainDataResize, numIter)
mnist.visualize(means:reshape(K, 1, 32, 32))

-- local means = torch.FloatTensor(numClasses, 1024)
-- for k = 1,numClasses do
--     local N = trainDataResize:size()[1]
--     local kArray = torch.ByteTensor(N):zero() + k
--     local trainSub = trainDataResize[train.labels:eq(kArray):reshape(N, 1):expand(N, 1024)]
--     trainSub = trainSub:reshape(trainSub:numel() / 1024, 1024)
--     means[k] = trainSub:mean(1)
-- end
-- mnist.visualize(means:reshape(numClasses, 1, 32, 32))

-- local newLabels = torch.range(1, 10)
-- local knn = require('nearest_neighbours')
-- local numTest = 100
-- local testPred = knn.runAll(1, means, newLabels, testDataResize, numTest)
-- local testLabelsSubset = test.labels:index(1, torch.range(1, numTest):long())
-- local testDataSubset = test.data:index(1, torch.range(1, numTest):long()) 
-- utils.evalPrediction(testPred, testLabelsSubset)
-- local neqIdx = testPred:ne(testLabelsSubset)
-- local neqIdx = neqIdx:reshape(numTest, 1, 1, 1):expand(numTest, 1, 32, 32)
-- local neqExamples = testDataSubset[neqIdx]
-- if neqExamples:numel() > 0 then
--     neqExamples = neqExamples:reshape(neqExamples:numel() / 1024, 1, 32, 32)
--     mnist.visualize(neqExamples)
-- end
