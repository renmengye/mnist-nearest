local torch = require('torch')
local Logger = require('logger')
local logger = Logger()
local utils = {}

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
