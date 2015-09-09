local nn = require('nn')
local nntrainer = require('nntrainer')
local hdf5 = require('hdf5')
local utils = require('utils')
local Logger = require('logger')
local logger = Logger()

torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

function createModel()
    model = nn.Sequential()
    model:add(nn.Linear(4596, 431))
    logger:logInfo('Created model')
    print(model)
    return model
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('ImageQA Linear Training')
cmd:text()
cmd:text('Options:')
cmd:option('-train', false, 'whether to train a new network')
cmd:option('-path', 'imageqa_linear.w', 'save network path')
cmd:option('-save', false, 'whether to save the trained network')
cmd:option('-norm', false, 'whether to have the normalized feature')
cmd:text()
opt = cmd:parse(arg)

local dataPath
if opt.norm then
-- local dataPath = '/ais/gobi3/u/mren/data/cocoqa-nearest/all.h5'
    dataPath = '../../data/cocoqa-nearest/all.h5'
else
    dataPath = '../../data/cocoqa-nearest/all_raw_2.h5'
end
local data = hdf5.open(dataPath, 'r'):read('answer'):all()

local imgbow = '../../data/img_bow.h5'
local init_weights = hdf5.open(imgbow, 'r'):read('answer'):all()
print(init_weights:size())

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
    local weights = model:getParameters()
    -- weights:copy(torch.rand(weights:size()) * 0.01 - 0.005)
    -- local weights = model:getParameters()
    -- weights:copy(torch.rand(weights:size()) * 0.001 - 0.0005)
    weights:copy(init_weights)
    local trainLabel = data.trainLabel + 1
    local validLabel = data.validLabel + 1
    local testLabel = data.testLabel + 1
    local trainPlusValidData = torch.cat(data.trainData, data.validData, 1)
    local trainPlusValidLabel = torch.cat(trainLabel, validLabel, 1)
    -- nntrainer.trainAll(
    --     model, trainPlusValidData, trainPlusValidLabel, data.testData, testLabel, 
    --     loopConfig, optimizer, optimConfig)
    -- if opt.save then
    --     nntrainer.save(opt.path, model)
    -- end
    local rate = nntrainer.evaluate(model, data.testData, testLabel, 100)
end
