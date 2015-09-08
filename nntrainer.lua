local torch = require('torch')
local optim = require('optim')
local Logger = require('logger')
local logger = Logger()
local nntrainer = {}

----------------------------------------------------------------------
function nntrainer.forwardOnce(model, x, labels)
   local criterion = nn.CrossEntropyCriterion()
   local pred = model:forward(x)
   local err = criterion:forward(pred, labels)
   local predScore, predClass = pred:max(2)
   predClass = predClass:reshape(labels:numel())
   local correct = predClass:eq(labels):sum()
   return err, correct, predClass, predScore
end

----------------------------------------------------------------------
function nntrainer.getEval(model, x, labels, w, dl_dw)
    local feval = function(w_new)
        if w ~= w_new then
            w:copy(w_new)
        end
        dl_dw:zero()
        local criterion = nn.CrossEntropyCriterion()
        local pred = model:forward(x)
        local loss = criterion:forward(pred, labels)
        model:backward(x, criterion:backward(pred, labels))
        return loss, dl_dw
    end
    return feval
end

----------------------------------------------------------------------
function nntrainer.getBatch(data, labels, batchSize, batchId)
    local numData = data:size()[1]
    local startIdx = batchSize * (batchId - 1) + 1
    local endIdx = batchSize * batchId
    if endIdx > numData then
        endIdx = numData
    end
    local xBatch = data:index(1, torch.range(startIdx, endIdx):long())
    local labelBatch = labels:index(1, torch.range(startIdx, endIdx):long())
    return xBatch, labelBatch
end

----------------------------------------------------------------------
function nntrainer.getBatchIter(N, batchSize)
    return torch.ceil(N / batchSize)
end

----------------------------------------------------------------------
function nntrainer.trainEpoch(model, data, labels, batchSize, w, dl_dw, optimizer, optimConfig, state)
    local epochCost = 0
    local N = data:size()[1]
    model:training()
    for step = 1,nntrainer.getBatchIter(N, batchSize) do
        xBatch, labelBatch = nntrainer.getBatch(data, labels, batchSize, step)
        _, cost = optimizer(
            nntrainer.getEval(model, xBatch, labelBatch, w, dl_dw), 
            w, optimConfig, state)
        epochCost = epochCost + cost[1] / xBatch:size()[1]
    end
    return epochCost
end

----------------------------------------------------------------------
function nntrainer.evaluate(model, data, labels, batchSize)
    local N = data:size()[1]
    local correct = 0
    model:evaluate()
    for step=1,nntrainer.getBatchIter(N, batchSize) do
        xBatch, labelBatch = nntrainer.getBatch(data, labels, batchSize, step)
        local _, correctBatch = nntrainer.forwardOnce(model, xBatch, labelBatch)
        correct = correct + correctBatch
    end
    return correct / N
end

----------------------------------------------------------------------
function nntrainer.trainAll(model, trainData, trainLabels, testData, testLabels, loopConfig, optimizer, optimConfig)
    local state = {}
    local w, dl_dw = model:getParameters()
    for epoch=1,loopConfig.numEpoch do
        trainLoss = nntrainer.trainEpoch(
            model, trainData, trainLabels, loopConfig.trainBatchSize,
            w, dl_dw, optimizer, optimConfig, state)
        trainRate = nntrainer.evaluate(
            model, trainData, trainLabels, loopConfig.evalBatchSize)
        testRate = nntrainer.evaluate(
            model, testData, testLabels, loopConfig.evalBatchSize)
        logger:logInfo(string.format(
            'n: %-2d l: %-6.3f tr: %.3f hr: %.3f', 
            epoch, trainLoss, trainRate, testRate))
    end
end

----------------------------------------------------------------------
function nntrainer.save(path, model)
    logger:logInfo(string.format('Saving model to %s', path))
    local w = model:getParameters()
    torch.save(path, w)
end

----------------------------------------------------------------------
function nntrainer.load(path, model)
    logger:logInfo(string.format('Loading model from %s', path))
    local w = model:getParameters()
    w:copy(torch.load(path))
    return model
end

----------------------------------------------------------------------
return nntrainer
