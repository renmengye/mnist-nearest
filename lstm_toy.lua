local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local rnn = require('rnn')
local lstm = require('lstm')
local constant = require('constant')
local batch_reshape = require('batch_reshape')
local lazy_sequential = require('lazy_sequential')
local lazy_gmodule = require('lazy_gmodule')
local logger = require('logger')()
local utils = require('utils')
local nntrainer = require('nntrainer')
local nnevaluator = require('nnevaluator')
local nnserializer = require('nnserializer')
local synthqa = require('synthqa')
local optim_pkg = require('optim')
local adam2 = require('adam')
torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')
-- torch.setdefaulttensortype('torch.DoubleTensor')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('LSTM Toy Training')
cmd:text()
cmd:text('Options:')
cmd:option('-task', 'parity', 'parity or sum')
cmd:option('-path', 'lstm_parity.w.h5', 'save network path')
cmd:option('-save', false, 'whether to save the trained network')
cmd:text()
opt = cmd:parse(arg)

-------------------------------------------------------------------------------
function createModel(numHidden, timespan)
    local numInput = 1
    local modelInput = nn.Identity()()
    local splitTable = nn.SplitTable(2)(modelInput)
    local constState = mynn.Constant({2 * numHidden}, 0)(modelInput)
    -- local lstmCore = lstm.createUnit(numInput, numHidden)
    -- local lstm = nn.RNN(lstmCore, timespan)({splitTable, constState})
    local lstm = lstm.createLayer(numInput, numHidden, timespan)(
        {splitTable, constState})
    local join = nn.JoinTable(2)(lstm)
    local lstmOutputReshape = mynn.BatchReshape(2 * numHidden)(join)
    local lstmOutputSel = nn.Narrow(
        2, numHidden + 1, numHidden)(lstmOutputReshape)
    local linear = nn.Linear(numHidden, 1)(lstmOutputSel)
    local final
    if opt.task == 'parity' then
        final = nn.Sigmoid()(linear)
    else
        final = linear
    end

    local all = nn.LazyGModule({modelInput}, {final})
    if opt.task == 'parity' then
        all.criterion = nn.BCECriterion()
        all.decision = function (pred)
            return torch.gt(pred, 0.5):float()
        end
    elseif opt.task == 'sum' then
        all.criterion = nn.MSECriterion()
        all.decision = function (pred)
            return torch.round(pred):float()
        end
    end
    all:addModule('lstm', lstm)
    all:addModule('linear', linear)
    all:setup()all:expand()
    return all
end

function getData(N, T)
    -- Parity --
    local data = torch.Tensor(N, T, 1)
    local label = torch.Tensor(N, T)
    for i = 1, N do
        if opt.task == 'sum' then
            local count = 0
            local attention = torch.floor(torch.uniform() * 2)
            data[i][1][1] = attention
            label[i][1] = 0
            for t = 2, T do
                data[i][t][1] = torch.floor(torch.uniform() * 2)
                if data[i][t][1] == attention then
                    count = count + 1
                end
                label[i][t] = count
            end
        elseif opt.task == 'parity' then
            local count = 0
            for t = 1, T do
                data[i][t][1] = torch.floor(torch.uniform() * 2)
                count = count + data[i][t][1]
                label[i][t] = count % 2
            end
        end
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
local H = 10
local T = 20
local params

local data, labels, rawData, model, evalModel, labelStart, classes
data = getData(N, T)
model = createModel(H, T)
evalModel = createModel(H, T)
if opt == 'sum' then
    labelStart = 0
    classes = {}
    for i = 1, T do
        classes[i] = string.format('%d', i)
    end
else
    labelStart = 1
    classes = {'0', '1'}
end

local gradClipTable = {
    lstm = 0.1,
    linear = 0.1
}

local loopConfig = {
    numEpoch = 1000,
    batchSize = 20,
    progressBar = false
}

local optimizer = optim.adam2
local optimConfig = {
    learningRate = 0.1,
    gradientClip = utils.gradientClip(1.0)
}

local nntrainer = NNTrainer(model, loopConfig, optimizer, optimConfig)
local trainEval = NNEvaluator('train', model)
local testEval = NNEvaluator('test', model)
local testSequenceLength = 25
local evalModel2 = createModel(H, testSequenceLength)
local visualizeSeq = function()
    local numItems = 1
    local T = testSequenceLength
    local testSubset
    if opt.task == 'parity' or opt.task == 'sum' then
        testSubset = getData(numItems * 2, T)
        evalModel2:getParameters():copy(model:getParameters())
        local y = evalModel2:forward(testSubset.testData)
        for n = 1, numItems do
            print('example', n)
            for t = 1, T do
                io.write(string.format('%6d', testSubset.testData[n][t][1]))
                io.write(string.format('%6.2f', y[t][1]))
                io.write(string.format('%6.2f', testSubset.testLabels[1][t]))
                io.write('\n')
            end
            io.write('\n')
        end
    end
end

visualizeSeq()
nntrainer:trainLoop(
    data.trainData, data.trainLabels,
    function(epoch)
        if epoch % 1 == 0 then
            trainEval:evaluate(data.trainData, data.trainLabels)
            testEval:evaluate(data.testData, data.testLabels)
        end
        if epoch % 10 == 0 then
            visualizeSeq()
        end
        if epoch % 100 == 0 and opt.save then
            logger:logInfo('saving model')
            nnserializer.save(model, opt.path)
        end
    end)

if opt.save then
    nnserializer.save(model, opt.path)
end
