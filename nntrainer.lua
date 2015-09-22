local torch = require('torch')
local optim = require('optim')
local logger = require('logger')()
local utils = require('utils')
local progress = require('progress_bar')
local nnserializer = require('nnserializer')
local nnevaluator = require('nnevaluator')

-------------------------------------------------------------------------------
local NNTrainerClass = torch.class('NNTrainer')

-------------------------------------------------------------------------------
function NNTrainer:__init(model, loopConfig, optimizer, optimConfig, cuda)
    self.model = model
    self.loopConfig = loopConfig
    self.optimizer = optimizer
    self.optimConfig = optimConfig
    if cuda then
        require('cutorch')
        require('cunn')
        local m = nn.Sequential()
        m.add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
        m.add(model:cuda())
        m.add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
        model = m
    end
    local state = {}
    local w, dl_dw = model:getParameters()
    model.w = w
    model.dl_dw = dl_dw
    self.runOptimizer = function(xBatch, labelBatch)
        return optimizer(
            self:getEvalFn(xBatch, labelBatch), w, optimConfig, state)
    end
    self.trainEvaluator = NNEvaluator('train', model, loopConfig.analyzers)
    self.testEvaluator = NNEvaluator('test', model, loopConfig.analyzers)
end

-------------------------------------------------------------------------------
function NNTrainer:getEvalFn(x, labels)
    local w = self.model.w
    local dl_dw = self.model.dl_dw
    local feval = function(w_new)
        if w ~= w_new then
            w:copy(w_new)
        end
        dl_dw:zero()
        local pred = self.model:forward(x)
        local loss = self.model.criterion:forward(pred, labels)
        self.model:backward(x, self.model.criterion:backward(pred, labels))
        
        logger:logInfo(
            string.format('Before clip: %.4f', torch.norm(dl_dw)), 2)
        if self.optimConfig.gradientClip then
            dl_dw:copy(self.optimConfig.gradientClip(dl_dw))
        end
        logger:logInfo(string.format('After clip: %.4f', torch.norm(dl_dw)), 2)
        return loss, dl_dw
    end
    return feval
end

-------------------------------------------------------------------------------
function NNTrainer:trainEpoch(data, labels, batchSize)
    self.model:training()
    for xBatch, labelBatch in utils.getBatchIterator(
        data, labels, batchSize, self.loopConfig.progressBar) do

        -- logger:logError('Before')
        -- local w1, dl_dw1 = self.model.moduleMap['lstm'].core:parameters()
        -- local replica = self.model.moduleMap['lstm'].replicas[1]
        -- local replica2 = self.model.moduleMap['lstm'].replicas[3]
        -- local w2, dl_dw2 = replica:parameters()
        -- local w3, dl_dw3 = replica2:parameters()
        -- for i = 1, #w1 do
        --     logger:logError(i)
        --     logger:logError(w1[i]:data())
        --     logger:logError(w2[i]:data())
        --     logger:logError(w3[i]:data())
        --     logger:logError(dl_dw1[i]:data())
        --     logger:logError(dl_dw2[i]:data())
        --     logger:logError(dl_dw3[i]:data())
        -- end
        -- print(self.model.dl_dw)

        self.runOptimizer(xBatch, labelBatch)
        collectgarbage()

        -- logger:logError('After')
        -- w1, dl_dw1 = self.model.rnn.core:parameters()
        -- replica = self.model.rnn.replicas[1]
        -- replica2 = self.model.rnn.replicas[3]
        -- w2, dl_dw2 = replica:parameters()
        -- w3, dl_dw3 = replica2:parameters()
        -- for i = 1, #w1 do
        --     logger:logError(i)
        --     logger:logError(w1[i]:data())
        --     logger:logError(w2[i]:data())
        --     logger:logError(w3[i]:data())
        --     logger:logError(dl_dw1[i]:data())
        --     logger:logError(dl_dw2[i]:data())
        --     logger:logError(dl_dw3[i]:data())
        -- end
    end
end

-------------------------------------------------------------------------------
function NNTrainer:checkGrad(data, labels)
    local evalFn = self:getEvalFn(data, labels)
    local dl_dw = torch.Tensor(self.model.dl_dw:size())
    local eps = 1e-7  -- This will only work in DoubleTensor
    local cost, dcost
    for i = 1, self.model.w:numel() do
        self.model.w[i] = self.model.w[i] + eps
        cost, dcost = evalFn(self.model.w)
        local dl_dw_tmp1 = cost
        self.model.w[i] = self.model.w[i] - 2 * eps
        cost, dcost = evalFn(self.model.w)
        local dl_dw_tmp2 = cost
        dl_dw[i] = (dl_dw_tmp1 - dl_dw_tmp2) / eps / 2
    end
    cost, dcost = evalFn(self.model.w)
    print(cost)
    logger:logError(dl_dw:cdiv(self.model.dl_dw))
end

-------------------------------------------------------------------------------
function NNTrainer:trainLoop(trainData, trainLabels, testData, testLabels, evaluators)
    for epoch = 1, self.loopConfig.numEpoch do
        self:trainEpoch(trainData, trainLabels, self.loopConfig.trainBatchSize)
        logger:logInfo(string.format('epoch: %-2d', epoch))
        self.trainEvaluator:evaluate(
            trainData, trainLabels, self.loopConfig.evalBatchSize)
        self.testEvaluator:evaluate(
            testData, testLabels, self.loopConfig.evalBatchSize)
        if self.loopConfig.savePath then
            nnserializer.save(self.model, savePath)
        end
    end
end
-------------------------------------------------------------------------------

----------------------------------------------------------------------
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
----- Add gradient clipping here
function nntrainer.getEval(model, x, labels, gradientClip)
    local w = model.w
    local dl_dw = model.dl_dw
    local feval = function(w_new)
        if w ~= w_new then
            w:copy(w_new)
        end
        dl_dw:zero()
        local criterion = nn.CrossEntropyCriterion()
        local pred = model:forward(x)
        
        local loss = criterion:forward(pred, labels)
        model:backward(x, criterion:backward(pred, labels))
        
        logger:logInfo(string.format('Before clip: %.4f', torch.norm(dl_dw)), 2)
        if gradientClip then
            dl_dw:copy(gradientClip(dl_dw))
        end
        logger:logInfo(string.format('After clip: %.4f', torch.norm(dl_dw)), 2)
        return loss, dl_dw
    end
    return feval
end

----------------------------------------------------------------------
function nntrainer.getBatchIterator(data, labels, batchSize, printProgress)
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

----------------------------------------------------------------------
function nntrainer.trainEpoch(model, data, labels, batchSize, runOptimizer)
    local epochCost = 0
    local N = data:size()[1]
    model:training()
    for xBatch, labelBatch in nntrainer.getBatchIterator(
        data, labels, batchSize) do
        _, cost = runOptimizer(xBatch, labelBatch)
        epochCost = epochCost + cost[1] * xBatch:size(1) / data:size(1)
        collectgarbage()
    end
    return epochCost
end

----------------------------------------------------------------------
function nntrainer.evaluate(model, data, labels, batchSize)
    local N = data:size()[1]
    local correct = 0
    model:evaluate()
    for xBatch, labelBatch in nntrainer.getBatchIterator(
            data, labels, batchSize, false) do
        local _, correctBatch = nntrainer.forwardOnce(
            model, xBatch, labelBatch)
        correct = correct + correctBatch
    end
    return correct / N
end

----------------------------------------------------------------------
function nntrainer.trainAll(model, trainData, trainLabels, testData, 
    testLabels, loopConfig, optimizer, optimConfig, cuda)
    if cuda then
        require('cutorch')
        require('cunn')
        local m = nn.Sequential()
        m.add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
        m.add(model:cuda())
        m.add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
        model = m
    end
    local state = {}
    local w, dl_dw = model:getParameters()
    model.w = w
    model.dl_dw = dl_dw
    local runOptimizer = function(xBatch, labelBatch)
        return optimizer(
            nntrainer.getEval(model, xBatch, labelBatch, 
                optimConfig.gradientClip), model.w, optimConfig, state)
    end
    for epoch = 1, loopConfig.numEpoch do
        trainLoss = nntrainer.trainEpoch(
            model, trainData, trainLabels, loopConfig.trainBatchSize, 
            runOptimizer)
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
return NNTrainer
