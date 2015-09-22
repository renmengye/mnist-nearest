local torch = require('torch')
local nn = require('nn')
local gnuplot = require('gnuplot')
local mnist = require('mnist')
local optim = require('optim')
local nntrainer = require('nntrainer')
local nnevaluator = require('nnevaluator')
local logger = require('logger')()
local nnserializer = require('nnserializer')

torch.manualSeed(2)
-- torch.setdefaulttensortype('torch.DoubleTensor')
torch.setdefaulttensortype('torch.FloatTensor')

function createModel()
    local model = nn.Sequential()
    model:add(nn.Linear(1024, 10))
    model.criterion = nn.CrossEntropyCriterion()
    model.decision = function(pred)
        local score, idx = pred:max(2)
        return idx
    end
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
    local trainer = NNTrainer(model, loopConfig, optimizer, optimConfig)
    -- trainer:checkGrad(trainData, trainLabels)
    trainer:trainLoop(trainData, trainLabels, testData, testLabels)
    if opt.save then
        nnserializer.save(model, opt.path)
    end
else
    local model = createModel()
    nnserializer.load(model, opt.path)
    local evaluator = NNEvaluator('test', model)
    evaluator:evaluate(testData, testLabels, 1000)
end
