local utils = require('utils')
local logger = require('logger')()
local nn = require('nn')

-------------------------------------------------------------------------------
local NNEvaluatorClass = torch.class('NNEvaluator')

function NNEvaluator.getClassConfusionAnalyzer(decision, classes)
    return function(pred, labels)
        local N = labels:numel()
        local labelDist = torch.histc(
            labels:double(), #classes, 1, #classes)
        local output = decision(pred):reshape(N)
        local confusion = torch.Tensor(#classes, #classes):zero()
        for n = 1, N do
            confusion[labels[n]][output[n]] = 
                confusion[labels[n]][output[n]] + 1
        end

        -- Percentage of class i being classified into class j
        logger:logInfo('--- Confusion matrix ---')
        io.write(string.format('%5s ', ''))
        for i = 1, #classes do
            if labelDist[i] > 0 then
                io.write(string.format('%5s ', classes[i]))
            end
        end
        io.write('\n')
        for i = 1, #classes do
            if labelDist[i] > 0 then
                io.write(string.format('%5s ', classes[i]))
                for j = 1, #classes do
                    if labelDist[j] > 0 then
                        io.write(string.format('%5.2f ', 
                            confusion[i][j] / labelDist[i]))
                    end
                end
                io.write('\n')
            end
        end
        io.flush()
    end
end

-------------------------------------------------------------------------------
function NNEvaluator.getClassAccuracyAnalyzer(decision, classes)
    return function (pred, labels)
        local N = labels:numel()
        local labelDist = torch.histc(
            labels:double(), #classes, 1, #classes)
        local output = decision(pred)
        local outputDist = torch.histc(
            output:double(), #classes, 1, #classes)
        local correct = output:eq(labels):reshape(N)
        local correctCls = {}
        for n = 1, N do
            if correct[n] == 1 then
                correctCls[#correctCls + 1] = labels[n]
            end
        end
        correctCls = torch.DoubleTensor(correctCls)
        local correctClsDist = torch.histc(
            correctCls, #classes, 1, #classes)
        for n = 1, #classes do
            if labelDist[n] > 0 then
                logger:logInfo(string.format(
                    '%s: %.3f', 
                    classes[n], correctClsDist[n] / labelDist[n]))
            end
        end
    end
end

-------------------------------------------------------------------------------
function NNEvaluator.getAccuracyAnalyzer(decision)
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
        if self.model.decision ~= nil then
            self.analyzers = {self.getAccuracyAnalyzer(self.model.decision)}
        else
            self.analyzers = {}
        end
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
