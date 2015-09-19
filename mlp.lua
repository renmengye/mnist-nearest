local torch = require('torch')
local nn = require('nn')
local gnuplot = require('gnuplot')
local mnist = require('mnist')
local optim = require('optim')
local NNTrainer = require('nntrainer')
local logger = require('logger')()

torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

function createModel()
    local model = nn.Sequential()
    -- model:add(nn.Linear(1024, 100))
    -- model:add(nn.ReLU())
    -- model:add(nn.Linear(100, 10))
    model:add(nn.Linear(1024, 10))
    logger:logInfo('Created model')
    print(model)
    return model
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST MLP Training')
cmd:text()
cmd:text('Options:')
cmd:option('-train', false, 'whether to train a new network')
cmd:option('-path', 'mnist_mlp.w', 'save network path')
cmd:option('-save', false, 'whether to save the trained network')
cmd:text()
opt = cmd:parse(arg)

local train, test = mnist.loadData()
local trainData, trainMean, trainStd = mnist.flattenFloatNormalize(train.data)
local testData = mnist.flattenFloatNormalize(test.data, trainMean, trainStd)
local trainLabels = train.labels:long()
local testLabels = test.labels:long()

if opt.train then
    local loopConfig = {
        numEpoch = 15,
        trainBatchSize = 20,
        evalBatchSize = 1000
    }
    local optimConfig = {
        learningRate = 0.01,
        momentum = 0.9
    }
    local optimizer = optim.sgd
    local model = createModel()
    model.criterion = nn.CrossEntropyCriterion()
    model.decision = function(prediction)
        local score, idx = prediction:max(2)
        return idx
    end
    local trainer = NNTrainer(model, loopConfig, optimizer, optimConfig)
    trainer:trainLoop(trainData, trainLabels, testData, testLabels)
    -- nntrainer.trainAll(
    --     model, trainData, trainLabels, testData, testLabels, 
    --     loopConfig, optimizer, optimConfig)
    -- if opt.save then
    --     nntrainer.save(opt.path, model)
    -- end
-- else
--     local model = nntrainer.load(opt.path, createModel())
--     local weight = model:get(1).weight
--     mnist.visualize(weight:reshape(100, 1, 32, 32))
--     local rate = nntrainer.evaluate(model, testData, testLabels, 1000)
--     logger:logInfo(string.format('Test rate: %.3f', rate))
end
