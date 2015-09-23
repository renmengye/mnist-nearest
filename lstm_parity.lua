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
local nntrainer = require('nntrainer')
local nnevaluator = require('nnevaluator')
local nnserializer = require('nnserializer')
local optim_pkg = require('optim')
torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')
-- torch.setdefaulttensortype('torch.DoubleTensor')

-------------------------------------------------------------------------------
function createModel(numHidden, timespan)
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
    all.criterion = nn.BCECriterion()
    all.decision = function (pred)
        return torch.gt(pred, 0.5):float()
    end
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
    numEpoch = 100,
    batchSize = 20,
    progressBar = false
}

local optimizer = optim.sgd
local model = createModel(H, T)

local optimConfig = {
    learningRate = 1.0,
    learningRates = utils.fillVector(
        torch.Tensor(model.w:size()), model.sliceLayer, learningRates),
    momentum = 0.9,
    gradientClip = utils.gradientClip(gradClipTable, model.sliceLayer)
}

local savePath = 'lstm_parity.w.h5'
local nntrainer = NNTrainer(model, loopConfig, optimizer, optimConfig)
local trainEval = NNEvaluator('train', model)
local testEval = NNEvaluator('test', model)
nntrainer:trainLoop(
    data.trainData, data.trainLabels,
    function(epoch)
        if epoch % 10 == 0 then
            trainEval:evaluate(data.trainData, data.trainLabels)
            testEval:evaluate(data.testData, data.testLabels)
        end
        if epoch % 100 == 0 then
            logger:logInfo('saving model')
            nnserializer.save(model, savePath)
        end
    end
    )

logger:logInfo('loading model2')
local model2 = createModel(H, T)
local nnevaluator = NNEvaluator('test', model2)
nnserializer.load(model2, 'lstm_parity.w.h5')
nnevaluator:evaluate(data.testData, data.testLabels, 1000)
