local torch = require('torch')
local Logger = require('logger')
local logger = Logger()
local progress = require('progress_bar')
local nearest_neighbours = {}

----------------------------------------------------------------------
function nearest_neighbours.distance(a, b)
    local d = (a - b):float()
    return d:cmul(d):sum()
end

----------------------------------------------------------------------
function nearest_neighbours.distanceBatch(A, b)
    local B, D
    B = b:reshape(1, b:size()[1])
    B = B:expand(A:size()[1], b:size()[1])
    D = A - B
    D = D:cmul(D):sum(2)
    return D
end

----------------------------------------------------------------------
function nearest_neighbours.consensus(labels, labelStart, labelEnd)
    local labelBins = torch.histc(
        labels:float(), labelEnd - labelStart + 1, labelStart, labelEnd)
    --print(labelBins)
    local maxBin, maxBinIdx = labelBins:max(1)
    logger:logInfo(string.format('Max bin number items: %d', maxBin[1]), 2)
    return maxBinIdx
end

----------------------------------------------------------------------
function nearest_neighbours.runOnce(data, labels, labelStart, labelEnd, example, k)
    local N = data:size()[1]
    local dist = 0
    local distAll = nearest_neighbours.distanceBatch(data, example)
    local distSort, idxSort = torch.sort(distAll, 1)
    local idxSortK = idxSort:index(1, torch.range(1, k):long()):reshape(k)
    local pred = nearest_neighbours.consensus(
        labels:index(1, idxSortK), labelStart, labelEnd)
    pred = pred - 1 + labelStart
    return pred, idxSortK
end

----------------------------------------------------------------------
function nearest_neighbours.runAll(K, trainData, trainLabels, testData, numTest, printProgress, processNearest)
    if printProgress == nil then
        printProgress = true
    end
    logger:logInfo(string.format('Running nearest neighbours, k=%d', K))
    logger:logInfo(string.format('Running %d test examples', numTest))
    local prediction = torch.LongTensor(numTest)
    local labelStart = trainLabels:min()
    local labelEnd = trainLabels:max()
    local progressBar = progress.get(numTest)
    logger:logInfo(string.format('Label start: %d', labelStart), 1)
    logger:logInfo(string.format('Label end: %d', labelEnd), 1)
    for i = 1,numTest do
        prediction[i], idxSortK = nearest_neighbours.runOnce(
            trainData, trainLabels, labelStart, labelEnd, testData[i], K, 
            processNearest)
        collectgarbage()
        if printProgress then
            progressBar(i)
        end
        if processNearest ~= nil then
            processNearest(i, idxSortK)
        end
    end
    return prediction
end

return nearest_neighbours
