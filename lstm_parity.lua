local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local rnn = require('rnn')
local lstm = require('lstm')
local constant = require('constant')
local lazy_sequential = require('lazy_sequential')
local lazy_gmodule = require('lazy_gmodule')
local logger = require('logger')()
local utils = require('utils')
torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')
-- torch.setdefaulttensortype('torch.DoubleTensor')

-------------------------------------------------------------------------------
function createModel(numHidden, timespan)
    -- The following code uses Sequential, which is less nice
    -- local numInput = 1
    -- local modelInput = nn.Identity()()
    -- local splitTable = nn.SplitTable(2)(modelInput)
    -- local constState = nn.Constant({2 * numHidden}, 0)(modelInput)
    -- local inputNode = nn.gModule({modelInput}, {splitTable, constState})

    -- local rnnCore = lstm.createUnit(numInput, numHidden)
    -- local rnn = nn.RNN(rnnCore, timespan)

    -- local lstmOutput = nn.Identity()()
    -- local lstmOutputTableSel = nn.SelectTable(timespan)(lstmOutput)
    -- local lstmOutputSel = nn.Narrow(
    --     2, numHidden + 1, numHidden)(lstmOutputTableSel)
    -- local linear = nn.Linear(numHidden, 1)(lstmOutputSel)
    -- local sigmoid = nn.Sigmoid()(linear)
    -- local aggregator = nn.gModule({lstmOutput}, {sigmoid})

    -- local all = nn.LazySequential()
    -- all:addModule('input', inputNode)
    -- all:addModule('lstm', rnn)
    -- all:addModule('sigmoid', aggregator)
    -- all:setup()
    -- rnn:expand()

    local numInput = 1
    local modelInput = nn.Identity()()
    local splitTable = nn.SplitTable(2)(modelInput)
    local constState = nn.Constant({2 * numHidden}, 0)(modelInput)
    local rnnCore = lstm.createUnit(numInput, numHidden)
    local rnn = nn.RNN(rnnCore, timespan)({splitTable, constState})
    local lstmOutputTableSel = nn.SelectTable(timespan)(rnn)
    local lstmOutputSel = nn.Narrow(
        2, numHidden + 1, numHidden)(lstmOutputTableSel)
    local linear = nn.Linear(numHidden, 1)(lstmOutputSel)
    local sigmoid = nn.Sigmoid()(linear)

    local all = nn.LazyGModule({modelInput}, {sigmoid})
    all:addModule('lstm', rnn)
    all:addModule('sigmoid', linear)
    all:setup()
    rnn.data.module:expand()
    return all
end

function getData(N, T)
    -- Parity --
    local data = torch.Tensor(N, T, 1)
    local label = torch.Tensor(N, 1)
    for i = 1, N do
        for t = 1, T do
            data[i][t][1] = torch.floor(torch.uniform() * 2)
        end
        label[i][1] = data[i]:sum() % 2
    end
    return {
        trainData = data[{{1, N / 2}}],
        trainLabels = label[{{1, N / 2}}],
        testData = data[{{N / 2 + 1, N}}],
        testLabels = label[{{N / 2 + 1, N}}],
    }
end

-- Generate train data
local N = 1000
local T = 8
local data = getData(N, T)
local H = 10
local gradClipTable = {
    lstm = 0.1,
    sigmoid = 0.1
}
local learningRates = {
    lstm = 0.8,
    sigmoid = 0.8
}

local loopConfig = {
    numEpoch = 10000,
    trainBatchSize = 20,
    evalBatchSize = 1000,
    progressBar = false
}

local NNTrainer = require('nntrainer')
local optim = require('optim')
local optimizer = optim.sgd
local model = createModel(H, T)
model.criterion = nn.BCECriterion()
model.decision = function (pred)
    return torch.gt(pred, 0.5):float()
end

local optimConfig = {
    learningRate = 1.0,
    learningRates = utils.fillVector(
        torch.Tensor(model.w:size()), model.sliceLayer, learningRates),
    momentum = 0.9,
    gradientClip = utils.gradientClip(gradClipTable, model.sliceLayer)
}

local trainer = NNTrainer(model, loopConfig, optimizer, optimConfig)
-- trainer:checkGrad(
--     data.trainData, data.trainLabels)
trainer:trainLoop(
    data.trainData, data.trainLabels, data.testData, data.testLabels)
