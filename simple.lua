local mnist = require('mnist')
local utils = require('utils')
local knn = require('nearest_neighbours')

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------
--local numTest = testSize[1]
local numTest = 200
local K = 5
local train, test = mnist.loadData()
local trainDataResize = mnist.flattenFloat(train.data)
local testDataResize = mnist.flattenFloat(test.data)
mnist.visualize(train.data)
mnist.visualize(test.data)
local testPred = knn.runAll(K, trainDataResize, train.labels, testDataResize, numTest)
local testLabelsSubset = test.labels:index(1, torch.range(1, numTest):long())
local testDataSubset = test.data:index(1, torch.range(1, numTest):long()) 
utils.evalPrediction(testPred, testLabelsSubset)
local neqIdx = testPred:ne(testLabelsSubset)
local neqIdx = neqIdx:reshape(numTest, 1, 1, 1):expand(numTest, 1, 32, 32)
local neqExamples = testDataSubset[neqIdx]
if neqExamples:numel() > 0 then
    neqExamples = neqExamples:reshape(neqExamples:numel() / 1024, 1, 32, 32)
    mnist.visualize(neqExamples)
end
