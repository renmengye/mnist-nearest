local torch = require('torch')
local utils = require('utils')
local knn = require('nearest_neighbours')
local kmeans = {}

function kmeans.run(K, data, numIter)
    -- Initialize
    local bagSize = 20
    local N = data:size()[1]
    local D = data:size()[2]
    local means = torch.FloatTensor(K, D)
    for k = 1,K do
        local bagIdx = utils.getBag(bagSize, N)
        means[k] = data:index(1, bagIdx):mean(1)
    end

    for i = 1,numIter do
        local distAll = torch.FloatTensor(K, N)
        for k = 1,K do
            distAll[k] = knn.distanceBatch(data, means[k])
        end
        local minDist, minIdx = distAll:min(1)
        local cost = 0
        for k = 1,K do
            local kArray = torch.LongTensor(N):zero() + k
            local kIdx = minIdx:eq(kArray):reshape(N, 1):expand(N, D)
            local dataSubset = data[kIdx]
            local m = dataSubset:size()[1] / 1024
            dataSubset = dataSubset:reshape(m, 1024)
            means[k] = dataSubset:mean(1)
            cost = cost + knn.distanceBatch(dataSubset, means[k]):sum() / m
        end
        print(string.format('iter: %d, cost: %.5f', i, cost))
        collectgarbage()
    end
    return means
end

return kmeans
