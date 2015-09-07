local knn = require('nearest_neighbours')
local hdf5 = require('hdf5')
local utils = require('utils')
local dataPath = '/ais/gobi3/u/mren/data/cocoqa-nearest/all.h5'
local data = hdf5.open(dataPath, 'r'):all()
-- for key,value in pairs(data) do 
--     print(key, value:size())
--     collectgarbage()
-- end

numTest = 100
for k = 1,21,5 do
    local validPred = knn.runAll(
        k, data.trainData, data.trainLabel, data.validData, numTest)
    local validLabelSubset = data.validLabel:index(1, torch.range(1, numTest):long())
    utils.evalPrediction(validPred, validLabelSubset)
end
