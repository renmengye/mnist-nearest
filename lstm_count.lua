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
torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('LSTM Seen Training')
cmd:text()
cmd:text('Options:')
-- cmd:option('-task', 'parity', 'parity or sum')
cmd:option('-path', 'lstm_seen.w.h5', 'save network path')
cmd:option('-save', false, 'whether to save the trained network')
cmd:option('-toy', 1, 'which toy to use')
cmd:option('-train', false, 'whether to train the network or just evaluate')
cmd:option('-num_ex', 10000, 'number of examples')
cmd:text()
opt = cmd:parse(arg)

function getData(N, T, M)
    -- Parity --
    local data = torch.Tensor(N, T, 3)
    local label = torch.Tensor(N, T)
    for i = 1, N do
        local set = {}
        for m = 1, M do
            set[m] = 0
        end
        for t = 1, T do
            local itemID = torch.floor(torch.uniform() * M) + 1
            data[i][t][1] = torch.floor((itemID - 1) / 4) % 2
            data[i][t][2] = torch.floor((itemID - 1) / 2) % 2
            data[i][t][3] = (itemID - 1) % 2
            label[i][t] = set[itemID]
            set[itemID] = set[itemID] + 1
        end
    end
    return {
        trainData = data[{{1, N / 2}}],
        trainLabels = label[{{1, N / 2}}],
        testData = data[{{N / 2 + 1, N}}],
        testLabels = label[{{N / 2 + 1, N}}],
    }
end

function createModel(params)
    -- params.timespan
    -- params.inputDim
    -- params.lstmDim
    local input = nn.Identity()()
    local inputSplit = nn.SplitTable(2)(input)
    local lstmCore = lstm.createUnit(params.inputDim, params.lstmDim)
    local lstmConstState = mynn.Constant(2 * params.lstmDim)(input)
    local lstm = nn.RNN(
        lstmCore, params.timespan)({inputSplit, lstmConstState})
    local lstmTimeJoin = mynn.BatchReshape(
        params.timespan, params.lstmDim * 2)(nn.JoinTable(2)(lstm))
    local lstmStateSel = nn.Narrow(
        3, params.lstmDim + 1, params.lstmDim)(lstmTimeJoin)
    local lstmStateReshape = mynn.BatchReshape(params.lstmDim)(lstmStateSel)
    local linear = nn.Linear(params.lstmDim, 1)(lstmStateReshape)
    local answerReshape = mynn.BatchReshape(params.timespan)(linear)
    local all = nn.LazyGModule({input}, {answerReshape})
    all:addModule('lstm', lstm)
    all:addModule('linear', linear)
    all:setup()
    lstm.data.module:expand()
    all.criterion = nn.MSECriterion()
    all.decision = function(pred)
        return torch.round(pred)
    end
    return all
end

function createModel2(params)
    -- params.numMem
    -- params.timespan
    -- params.inputDim
    -- params.memDim
    local input = nn.Identity()()
    local inputSplit = nn.SplitTable(2)(input)
    local memNetCore = lstm.createMemoryUnitWithBinaryOutput2(
        params.inputDim, params.numMem, params.memDim)
    local memNetConstState = mynn.Constant(
        params.numMem * params.memDim + 1)(input)
    local memNet = nn.RNN(
        memNetCore, params.timespan)({inputSplit, memNetConstState})
    local memNetOutputs = {}
    for t = 1, params.timespan do
        table.insert(memNetOutputs, nn.Narrow(
            2, params.numMem * params.memDim + 1, 1)(nn.SelectTable(t)(memNet)))
    end
    local memNetOutputsJoin = nn.JoinTable(2)(memNetOutputs)
    local all = nn.LazyGModule({input}, {memNetOutputsJoin})
    all:addModule('memNet', memNet)
    all:setup()
    memNet.data.module:expand()
    all.criterion = nn.BCECriterion()
    all.decision = function(pred)
        return pred:gt(0.5):float()
    end
    return all
end

function createModel3(params)
    -- params.numMem
    -- params.timespan
    -- params.inputDim
    -- params.memDim
    -- params.forget
    local input = nn.Identity()()
    local inputSplit = nn.SplitTable(2)(input)
    local memNetCore = lstm.createMemoryUnitWithBinaryOutput3(
        params.inputDim, params.numMem, params.memDim, params.forget)
    local memNetConstState = mynn.Constant(
        params.numMem * params.memDim + params.memDim + params.numMem)(input)
    local memNet = nn.RNN(
        memNetCore, params.timespan)({inputSplit, memNetConstState, })
    local memNetOutputs = {}
    for t = 1, params.timespan do
        table.insert(memNetOutputs, nn.Narrow(
            2, params.numMem * params.memDim + 1, params.memDim)(
            nn.SelectTable(t)(memNet)))
    end
    local memNetOutputsJoin = mynn.BatchReshape(params.memDim)(
        nn.JoinTable(2)(memNetOutputs))
    local linear = nn.Linear(params.memDim, 1)(memNetOutputsJoin)
    local answerReshape = mynn.BatchReshape(params.timespan)(linear)
    local all = nn.LazyGModule({input}, {answerReshape})
    linear.data.module.weight:uniform(0.1 / 2, 0.1 / 2)
    all:addModule('memNet', memNet)
    all:addModule('linear', linear)
    all:setup()
    memNet.data.module:expand()
    all.criterion = nn.MSECriterion()
    all.decision = function(pred)
        return torch.round(pred)
    end
    return all
end


-- Generate train data
local numEx = opt.num_ex
local lstmDim = 10
local timespan = 20
local numUniqueItems = 8
local inputDim = torch.log(numUniqueItems) / torch.log(2)
local numMem = 12
local memDim = 5
local rawData
local data

local params
local gradClipTable
local model
data = getData(numEx, timespan, numUniqueItems)
if opt.toy == 1 then
    params = {
        timespan = timespan,
        inputDim = inputDim,
        lstmDim = lstmDim
    }
    gradClipTable = {
        lstm = 0.1,
        linear = 0.1
    }
    model = createModel(params)
elseif opt.toy == 2 then
    params = {
        timespan = timespan,
        inputDim = inputDim,
        memDim = memDim,
        numMem = numMem
    }
    gradClipTable = {
        memNet = 0.1
    }
    model = createModel2(params)
elseif opt.toy == 3 then
    params = {
        timespan = timespan,
        inputDim = inputDim,
        memDim = memDim,
        numMem = numMem,
        forget = true
    }
    gradClipTable = {
        memNet = 0.1
    }
    model = createModel3(params)
end

local loopConfig = {
    numEpoch = 10000,
    batchSize = 20,
    progressBar = false
}

local optimConfig = {
    learningRate = 0.01,
    gradientClip = utils.gradientClip(gradClipTable, model.sliceLayer)
}

local optimizer = optim.adam
local savePath = 'seen.w.h5'
local nntrainer = NNTrainer(model, loopConfig, optimizer, optimConfig)
local trainEval = NNEvaluator('train', model)
local testEval = NNEvaluator('test', model)

local visualize = function()
    local output = model:forward(data.testData[{{1, 5}}])
    if opt.toy == 1 then
        for i = 1, 5 do
            local a = torch.cat(data.testData[i], output[i], 2)
            local b = torch.cat(a, data.testLabels[i], 2)
            print(b)
        end
    else
        for i = 1, 5 do
            print('test example, write head pointer')
            local memNet = model.moduleMap['memNet']
            for t = 1, timespan do
                for d = 1, inputDim do
                    io.write(string.format('%1d ', data.testData[i][t][d]))
                end
                io.write(' ')
                for d = 1, numMem do
                    io.write(string.format(
                        '%.2f ', 
                        memNet.replicas[t].moduleMap['writeHeadNorm'].output[i][d]))
                end
                io.write(string.format(' %.2f ', output[i][t]))
                io.write(string.format('%1d', data.testLabels[i][t]))
                io.write('\n')
            end
        end
    end
end

if opt.train then
    nntrainer:trainLoop(
        data.trainData, data.trainLabels,
        function(epoch)
            if epoch % 1 == 0 then
                trainEval:evaluate(data.trainData, data.trainLabels)
                testEval:evaluate(data.testData, data.testLabels)
            end
            if epoch % 20 == 0 then
                visualize()
                if opt.save then
                    logger:logInfo('saving model')
                    nnserializer.save(model, savePath)
                end
            end
        end
        )
end
