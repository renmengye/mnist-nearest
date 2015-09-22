local utils = require('utils')
local logger = require('logger')()
local nn = require('nn')

-------------------------------------------------------------------------------
local NNEvaluatorClass = torch.class('NNEvaluator')

-------------------------------------------------------------------------------
local function getAccuracyAnalyzer(decision)
    return function(pred, labels)
        local predClass = decision(pred)
        local correct = predClass:eq(labels):sum()
        logger:logInfo(string.format('rate: %.3f', correct / labels:numel()))
    end
end

-------------------------------------------------------------------------------
function NNEvaluator:__init(name, model, analyzers)
    self.name = name
    self.model = model
    if analyzers and type(analyzers) == 'table' then
        self.analyzers = analyzers
    elseif analyzers then
        logger:logError('type of analyzers must be table')
    else
        self.analyzers = {}
    end
    if self.model.decision ~= nil then
        table.insert(self.analyzers, getAccuracyAnalyzer(self.model.decision))
    end
end

-------------------------------------------------------------------------------
function NNEvaluator:forwardOnce(data, labels)
    local pred = self.model:forward(data)
    local loss = self.model.criterion:forward(pred, labels)
    return pred, loss
end

-------------------------------------------------------------------------------
function NNEvaluator:evaluate(data, labels, batchSize)
    if batchSize == nil then
        batchSize = 100
    end
    local N = data:size()[1]
    local epochPred = {}
    local epochLoss = 0
    self.model:evaluate()
    local printed = false
    for xBatch, labelBatch in utils.getBatchIterator(
            data, labels, batchSize, false) do
        local pred, loss = self:forwardOnce(xBatch, labelBatch)
        table.insert(epochPred, pred:clone())
        epochLoss = epochLoss + loss * xBatch:size(1) / data:size(1)
        collectgarbage()
    end
    logger:logInfo(string.format('running eval: %s', self.name))
    logger:logInfo(string.format('loss: %.3f', epochLoss))
    local allEpochPred = nn.JoinTable(1):forward(epochPred)
    for i, analyzer in ipairs(self.analyzers) do
        analyzer(allEpochPred, labels, self.model)
    end
end

return NNEvaluator
