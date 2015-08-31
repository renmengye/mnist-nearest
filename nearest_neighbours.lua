local torch = require('torch')
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
function nearest_neighbours.consensus(labels)
    local labelBins = torch.histc(labels:float(), 10, 1, 10)
    local maxBin, maxBinIdx = labelBins:max(1)
    return maxBinIdx
end

----------------------------------------------------------------------
function nearest_neighbours.runOnce(data, labels, example, k)
    local N = data:size()[1]
    local dist = 0
    local distAll = nearest_neighbours.distanceBatch(data, example)
    local distSort, idxSort = torch.sort(distAll, 1)
    -- print(distSort)
    -- print(idxSort)
    local idxSortK = idxSort:index(1, torch.range(1, k):long())
    local pred = nearest_neighbours.consensus(labels:index(1, idxSortK[1]))
    return pred
end

----------------------------------------------------------------------
function nearest_neighbours.runAll(K, trainData, trainLabels, testData, numTest)
    print(string.format('==> running nearest neighbours, k = %d', K))
    print(string.format('==> running %d test examples', numTest))
    local prediction = torch.ByteTensor(numTest)
    local progress = 0
    for i = 1,numTest do
        prediction[i] = nearest_neighbours.runOnce(trainData, trainLabels, testData[i], K)
        --print(string.format('Example: %d, Pred: %d, GT: %d', i, prediction[i], test.labels[i]))
        collectgarbage()
        while i / numTest > progress / 80 do
            io.write('.')
            io.flush()
            progress = progress + 1
        end
    end
    return prediction
end

return nearest_neighbours
