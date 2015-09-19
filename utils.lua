local torch = require('torch')
local logger = require('logger')()
local progress = require('progress_bar')
local utils = {}

---------------------------------------------------------------------
function utils.getBatchIterator(data, labels, batchSize, printProgress)
    if printProgress == nil then
        printProgress = true
    end
    local step = 0
    local numData = data:size()[1]
    local numSteps = torch.ceil(numData / batchSize)
    local progressBar = progress.get(numSteps)
    return function()
               if step < numSteps then
                   local startIdx = batchSize * step + 1
                   local endIdx = batchSize * (step + 1)
                   if endIdx > numData then
                       endIdx = numData
                   end
                   local xBatch = data:index(
                       1, torch.range(startIdx, endIdx):long())
                   local labelBatch = labels:index(
                       1, torch.range(startIdx, endIdx):long())
                   step = step + 1
                   if printProgress then
                       progressBar(step)
                   end
                   return xBatch, labelBatch
               end
           end
end

---------------------------------------------------------------------
function utils.evalPrediction(prediction, labels)
    local correct = prediction:eq(labels):int():sum()
    local rate = correct / prediction:size()[1]
    labels = labels:reshape(labels:numel()):long()
    prediction = prediction:reshape(prediction:numel()):long()
    logger:logInfo(string.format('Accuracy: %.5f', rate))
    for i = 1,labels:size()[1] do
        logger:logInfo(string.format('Label: %d, Pred: %d', labels[i], prediction[i]), 1)
    end
    return rate
end

---------------------------------------------------------------------
function utils.getBag(N, max)
    local r = torch.rand(N) * max
    local items = torch.ceil(r):long()
    return items
end

return utils
