local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local rnn = require('rnn')
local gru = require('gru')
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

-------------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('GRU Toy Training')
cmd:text()
cmd:text('Options:')
cmd:option('-task', 'parity', 'parity or sum')
cmd:option('-path', 'gru_parity.w.h5', 'save network path')
cmd:option('-save', false, 'whether to save the trained network')
cmd:text()
opt = cmd:parse(arg)

-------------------------------------------------------------------------------
function createModel(numHidden, timespan)
    local numInput = 1
    local modelInput = nn.Identity()()
    local splitTable = nn.SplitTable(2)(modelInput)
    local constState = mynn.Constant({numHidden}, 0)(modelInput)
    local gruCore = gru.createUnit(numInput, numHidden)
    local grnn = nn.RNN(gruCore, timespan)({splitTable, constState})
    local join = nn.JoinTable(2)(grnn)
    local grnnOutputReshape = mynn.BatchReshape(numHidden)(join)
    local linear = nn.Linear(numHidden, 1)(grnnOutputReshape)
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
    all:addModule('grnn', grnn)
    all:addModule('linear', linear)
    all:setup()
    grnn.data.module:expand()
    return all
end

-------------------------------------------------------------------------------
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

-------------------------------------------------------------------------------
local N = 1000
local H = 10
local T = 20
local TT = 25

-------------------------------------------------------------------------------
local data, labels, rawData, model, evalModel, evalModel2, labelStart, classes
data = getData(N, T)
model = createModel(H, T)
evalModel = createModel(H, T)
evalModel2 = createModel(H, TT)

-------------------------------------------------------------------------------
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
    grnn = 0.1,
    linear = 0.1
}
local loopConfig = {
    numEpoch = 1000,
    batchSize = 20,
    progressBar = false
}

-------------------------------------------------------------------------------
local optimizer = optim.adam2
local optimConfig = {
    learningRate = 0.1,
    gradientClip = utils.gradientClip(gradClipTable, model.sliceLayer)
}

-------------------------------------------------------------------------------
local nntrainer = NNTrainer(model, loopConfig, optimizer, optimConfig)
local trainEval = NNEvaluator('train', model)
local testEval = NNEvaluator('test', model)
local visualizeSeq = function()
    local numItems = 1
    local testSubset
    if opt.task == 'parity' or opt.task == 'sum' then
        testSubset = getData(numItems * 2, TT)
        evalModel2.w:copy(model.w)
        local y = evalModel2:forward(testSubset.testData)
        for n = 1, numItems do
            print('example', n)
            for t = 1, TT do
                io.write(string.format('%6d', testSubset.testData[n][t][1]))
                io.write(string.format('%6.2f', y[t][1]))
                io.write(string.format('%6.2f', testSubset.testLabels[1][t]))
                io.write('\n')
            end
            io.write('\n')
        end
    end
end

-------------------------------------------------------------------------------
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

-------------------------------------------------------------------------------
if opt.save then
    nnserializer.save(model, opt.path)
end
