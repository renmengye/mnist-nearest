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
    model:add(nn.LogSoftMax())
    model.criterion = nn.ClassNLLCriterion()
    -- model.criterion = nn.CrossEntropyCriterion()
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
local data = {}
local trainMean, trainStd
data.trainData, trainMean, trainStd =  mnist.flattenFloatNormalize(train.data)
data.testData = mnist.flattenFloatNormalize(test.data, trainMean, trainStd)
data.trainLabels = train.labels:long()
data.testLabels = test.labels:long()

local classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}
if opt.train then
    local loopConfig = {
        numEpoch = 15,
        batchSize = 20
    }
    local optimConfig = {
        learningRate = 0.01,
        momentum = 0.9
    }
    local optimizer = optim.sgd
    local model = createModel()
    local trainer = NNTrainer(model, loopConfig, optimizer, optimConfig)
    local trainEval = NNEvaluator('train', model)
    local testEval = NNEvaluator('test', model)
    local testClsEval = NNEvaluator('test class', model, {
        NNEvaluator.getClassAccuracyAnalyzer(
            model.decision, classes),
        NNEvaluator.getClassConfusionAnalyzer(
            model.decision, classes)})
    trainer:trainLoop(data.trainData, data.trainLabels, 
        function(epoch)
            trainEval:evaluate(data.trainData, data.trainLabels)
            testEval:evaluate(data.testData, data.testLabels)
            testClsEval:evaluate(data.testData, data.testLabels)
            if opt.save then
                nnserializer.save(model, opt.path)
            end
        end
        )
else
    local model = createModel()
    nnserializer.load(model, opt.path)
    local evaluator = NNEvaluator('test', model)
    evaluator:evaluate(testData, testLabels, 1000)
end
