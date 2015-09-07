local torch = require('torch')
local Logger = require('logger')
local logger = Logger('utils.lua', '')
local utils = {}

---------------------------------------------------------------------
function utils.evalPrediction(prediction, labels)
    local correct = prediction:eq(labels):int():sum()
    local rate = correct / prediction:size()[1]
    logger:logInfo(string.format('Accuracy: %.5f', rate))
    return rate
end

---------------------------------------------------------------------
function utils.getBag(N, max)
    local r = torch.rand(N) * max
    local items = torch.ceil(r):long()
    return items

end

return utils
