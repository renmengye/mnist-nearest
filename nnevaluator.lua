local utils = require('utils')
----------------------------------------------------------------------
local NNEvaluatorClass = torch.class('NNEvaluator')

----------------------------------------------------------------------
function NNEvaluator:__init(model)
    self.model = model
end

----------------------------------------------------------------------
function NNEvaluator:forwardOnce(data, labels)
    local pred = self.model:forward(data)
    local cost = self.model.criterion:forward(pred, labels)
    local correct
    if self.model.decision ~= nil then
        local predClass = self.model.decision(pred):reshape(labels:numel())
        correct = predClass:eq(labels):sum()
    else
        correct = 0
    end
    return cost, correct
end

----------------------------------------------------------------------
function NNEvaluator:evaluate(data, labels, batchSize)
    local N = data:size()[1]
    local epochRate = 0
    local epochCost = 0
    self.model:evaluate()
    for xBatch, labelBatch in utils.getBatchIterator(
            data, labels, batchSize, false) do
        local cost, correct = self:forwardOnce(xBatch, labelBatch)
        epochRate = epochRate + correct / N
        epochCost = epochCost + cost * xBatch:size(1) / data:size(1)
        collectgarbage()
    end
    return epochCost, epochRate
end

return NNEvaluator
