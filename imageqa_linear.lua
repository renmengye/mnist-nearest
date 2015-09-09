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
cmd:option('-normimg', false, 'whether to have the normalized image feature')
cmd:option('-normbow', false, 'whether to have the normalized bow feature')
cmd:text()
opt = cmd:parse(arg)

local dataPath
if opt.normimg and opt.normbow then
-- local dataPath = '/ais/gobi3/u/mren/data/cocoqa-nearest/all.h5'
    dataPath = '../../data/cocoqa-nearest/all_inorm_bnorm.h5'
elseif opt.normbow then
    dataPath = '../../data/cocoqa-nearest/all_iraw_bnorm.h5'
elseif opt.normimg then
    dataPath = '../../data/cocoqa-nearest/all_inorm_braw.h5'
else
    dataPath = '../../data/cocoqa-nearest/all_iraw_braw.h5'
end
print(dataPath)
local data = hdf5.open(dataPath, 'r'):all()
local imgbow = '../../data/img_bow.h5'
local init_weights = hdf5.open(imgbow, 'r'):all()
print(init_weights.answer:size())

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

    local weights = model:parameters()
    for key,value in pairs(weights) do
        print(key)
        print(value:size())
    end
    
    -- weights[1]:copy(init_weights.answer:index(1, torch.range(1, 4596):long()):t())
    -- weights[2]:copy(init_weights.answer[4597])

    local weights = model:getParameters()
    weights:copy(torch.rand(weights:size()) * 0.01 - 0.005)

    local trainLabel = data.trainLabel + 1
    local validLabel = data.validLabel + 1
    local testLabel = data.testLabel + 1
    local trainPlusValidData = torch.cat(data.trainData, data.validData, 1)
    local trainPlusValidLabel = torch.cat(trainLabel, validLabel, 1)
    nntrainer.trainAll(
        model, trainPlusValidData, trainPlusValidLabel, data.testData, testLabel, 
        loopConfig, optimizer, optimConfig)
    if opt.save then
        nntrainer.save(opt.path, model)
    end
    local rate = nntrainer.evaluate(model, data.testData, testLabel, 100)
    logger:logInfo(string.format('Accuracy: %.4f', rate))
end
