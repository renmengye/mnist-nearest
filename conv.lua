local torch = require('torch')
local nn = require('nn')
local gnuplot = require('gnuplot')
local mnist = require('mnist')
local optim = require('optim')
local nntrainer = require('nntrainer')
local Logger = require('logger')
local logger = Logger('conv.lua', '')
torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

function createModel()
    local model = nn.Sequential()
    model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
    model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    model:add(nn.View(64 * 2 * 2))
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(64 * 2 * 2, 100))
    model:add(nn.ReLU())
    model:add(nn.Linear(100, 10))
    logger:logInfo('Created model')
    print(model)
    return model
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST ConvNet Training')
cmd:text()
cmd:text('Options:')
cmd:option('-train', false, 'whether to train a new network')
cmd:option('-path', 'mnist_convnet.w', 'save network path')
cmd:option('-save', false, 'whether to save the trained network')
cmd:text()
opt = cmd:parse(arg)

local train, test = mnist.loadData()
local trainData, trainMean, trainStd = mnist.floatNormalize(train.data)
local testData = mnist.floatNormalize(test.data, trainMean, trainStd)
local trainLabels = train.labels:long()
local testLabels = test.labels:long()

if opt.train then
    local loopConfig = {
        numEpoch = 5,
        trainBatchSize = 20,
        evalBatchSize = 1000
    }
    local optimConfig = {
        learningRate = 0.01,
        momentum = 0.9
    }
    local optimizer = optim.sgd
    local model = createModel()
    nntrainer.trainAll(
        model, trainData, trainLabels, testData, testLabels, 
        loopConfig, optimizer, optimConfig)    
    if opt.save then
        nntrainer.save(opt.path, model)
    end
else
    local model = nntrainer.load(opt.path, createModel())
    local weight = model:get(1).weight
    mnist.visualize(weight:reshape(32, 1, 5, 5))
    local rate = nntrainer.evaluate(model, testData, testLabels, 1000)
    logger:logInfo(string.format('Test rate: %.3f', rate))
end
