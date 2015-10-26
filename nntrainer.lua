local torch = require('torch')
local optim = require('optim')
local logger = require('logger')()
local utils = require('utils')
local progress = require('progress_bar')

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
        local pred, pred2 = self.model:forward(x)
        local loss = self.model.criterion:forward(pred, labels)
        self.model:backward(x, self.model.criterion:backward(pred, labels))
        logger:logInfo(
            string.format('Before clip: %.4f', torch.norm(dl_dw)), 2)
        if self.optimConfig.gradientClip then
            dl_dw:copy(self.optimConfig.gradientClip(dl_dw))
        end
        -- print(dl_dw)
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
        -- print(self.model.sliceLayer(self.model.dl_dw, 'wordEmbed'))
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
function NNTrainer:trainLoop(trainData, trainLabels, dofn)
    for epoch = 1, self.loopConfig.numEpoch do
        self:trainEpoch(trainData, trainLabels, self.loopConfig.batchSize)
        logger:logInfo(string.format('epoch: %-2d', epoch))
        if dofn then
            dofn(epoch)
        end
    end
end
-------------------------------------------------------------------------------

return NNTrainer
