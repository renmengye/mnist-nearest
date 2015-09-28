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
cmd:option('-noise', false, 'whether to add noise')
cmd:option('-forget', false, 'whether to add forget gate')
cmd:option('-query_write', false, 'whether to add query write attention')
cmd:option('-synthqa', false, 'whether to have synthqa dataset')
cmd:text()
opt = cmd:parse(arg)

logger:logInfo('--- command line options ---')
for key, value in pairs(opt) do
    logger:logInfo(string.format('%s: %s', key, value))
end
logger:logInfo('----------------------------')

function getDataSynthQA(N, T, addNoise)
    local labels = torch.Tensor(N, T)
    local data = torch.Tensor(N, T, 2 + 2 + 4)
    local allItems = {}
    for i = 1, N do
        local set = {}
        for m = 1, 4 do
            set[m] = {}
            for n = 1, 4 do
                set[m][n] = {}
            end
        end
        local cat = torch.ceil(torch.uniform() * 4)
        local color = torch.ceil(torch.uniform() * 4)
        for t = 1, T do
            local item = {
                category = cat,
                color = color,
                grid = torch.ceil(torch.uniform() * 9)
            }
            table.insert(allItems, item)
            data[i][t][1] = torch.floor((item.category - 1) / 2) % 2
            data[i][t][2] = (item.category - 1) % 2
            -- print(item.category, data[i][t][1], data[i][t][2])
            data[i][t][3] = torch.floor((item.color - 1) / 2) % 2
            data[i][t][4] = (item.color - 1) % 2
            data[i][t][{{5, 8}}] = synthqa.getCoord(item.grid, addNoise)
            if set[item.category][item.color][item.grid] ~= nil then
                labels[i][t] = 0
            else
                labels[i][t] = 1
                set[item.category][item.color][item.grid] = true
            end
        end
    end
    return {
        trainData = data[{{1, N / 2}}],
        trainLabels = labels[{{1, N / 2}}],
        testData = data[{{N / 2 + 1, N}}],
        testLabels = labels[{{N / 2 + 1, N}}],
    }
end

function getData(N, T, M, addNoise)
    -- Parity --
    local data = torch.Tensor(N, T, 3)
    local label = torch.Tensor(N, T)
    for i = 1, N do
        local set = {}
        for m = 1, M do
            set[m] = false
        end
        for t = 1, T do
            local itemID = torch.floor(torch.uniform() * M) + 1
            data[i][t][1] = torch.floor((itemID - 1) / 4) % 2
            data[i][t][2] = torch.floor((itemID - 1) / 2) % 2
            data[i][t][3] = (itemID - 1) % 2
            if addNoise then
                data[i][t][1] = data[i][t][1] + torch.normal(0.0, 0.1)
                data[i][t][2] = data[i][t][2] + torch.normal(0.0, 0.1)
                data[i][t][3] = data[i][t][3] + torch.normal(0.0, 0.1)
            end
            if set[itemID] then
                label[i][t] = 1
            else
                label[i][t] = 0
            end
            set[itemID] = true
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
    local answer = nn.Sigmoid()(linear)
    local answerReshape = mynn.BatchReshape(params.timespan)(answer)
    local all = nn.LazyGModule({input}, {answerReshape})
    all:addModule('lstm', lstm)
    all:addModule('linear', linear)
    all:setup()
    lstm.data.module:expand()
    all.criterion = nn.BCECriterion()
    all.decision = function(pred)
        return pred:gt(0.5):float()
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
        params.numMem * params.memDim + params.memDim)(input)
    local memNet = nn.RNN(
        memNetCore, params.timespan)({inputSplit, memNetConstState})
    local memNetOutputs = {}
    for t = 1, params.timespan do
        table.insert(memNetOutputs, nn.Narrow(
            2, params.numMem * params.memDim + 1, params.memDim)(
            nn.SelectTable(t)(memNet)))
    end
    local memNetOutputsJoin = mynn.BatchReshape(params.memDim)(
        nn.JoinTable(2)(memNetOutputs))
    local linear = nn.Linear(params.memDim, 1)(memNetOutputsJoin)
    local answer = nn.Sigmoid()(linear)
    local answerReshape = mynn.BatchReshape(params.timespan)(answer)

    local all = nn.LazyGModule({input}, {answerReshape})
    all:addModule('memNet', memNet)
    all:addModule('linear', linear)
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
    -- params.hasQueryWrite
    -- params.forget
    local input = nn.Identity()()
    local inputSplit = nn.SplitTable(2)(input)
    local memNetCore = lstm.createMemoryUnitWithBinaryOutput3(
        params.inputDim, params.numMem, params.memDim, params.hasQueryWrite, params.forget)
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
    local answer = nn.Sigmoid()(linear)
    local answerReshape = mynn.BatchReshape(params.timespan)(answer)
    local all = nn.LazyGModule({input}, {answerReshape})
    all:addModule('memNet', memNet)
    all:addModule('linear', linear)
    all:setup()
    memNet.data.module:expand()
    all.criterion = nn.BCECriterion()
    all.decision = function(pred)
        return pred:gt(0.5):float()
    end
    return all
end

function createModel4(params)
    -- params.numMem
    -- params.timespan
    -- params.inputDim
    -- params.memDim
    -- params.hasQueryWrite
    -- params.forget
    local input = nn.Identity()()
    local inputSplit = nn.SplitTable(2)(input)
    local memNetCore = lstm.createMemoryUnitWithBinaryOutput4(
        params.numMem, params.memDim, params.hasQueryWrite, params.forget)
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
    local answer = nn.Sigmoid()(linear)
    local answerReshape = mynn.BatchReshape(params.timespan)(answer)
    local all = nn.LazyGModule({input}, {answerReshape})
    all:addModule('memNet', memNet)
    all:addModule('linear', linear)
    all:setup()
    memNet.data.module:expand()
    all.criterion = nn.BCECriterion()
    all.decision = function(pred)
        return pred:gt(0.5):float()
    end
    return all
end
-- Generate train data
local numEx = opt.num_ex
local lstmDim
local data
local timespan
local numMem = 12
local memDim = 5

if opt.synthqa then
    inputDim = 8
    timespan = 23
    lstmDim = 10
    numMem = 10
    memDim = 8
    data = getDataSynthQA(numEx, timespan, opt.noise)
else
    timespan = 20
    lstmDim = 10
    numMem = 12
    memDim = 5
    local numUniqueItems = 8
    inputDim =  torch.log(numUniqueItems) / torch.log(2)
    data = getData(numEx, timespan, numUniqueItems, opt.noise)
end

local params
local gradClipTable
local model
logger:logInfo('data')
print(torch.cat(data.testData[5], data.testLabels[5], 2))

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
        memNet = 0.1,
        linear = 0.1
    }
    model = createModel2(params)
elseif opt.toy == 3 then
    params = {
        timespan = timespan,
        inputDim = inputDim,
        memDim = memDim,
        numMem = numMem,
        forget = opt.forget,
        hasQueryWrite = opt.query_write
    }
    gradClipTable = {
        memNet = 0.1,
        linear = 0.1
    }
    model = createModel3(params)
elseif opt.toy == 4 then
    params = {
        timespan = timespan,
        memDim = inputDim,
        numMem = numMem,
        forget = opt.forget,
        hasQueryWrite = opt.query_write
    }
    gradClipTable = {
        memNet = 0.1,
        linear = 0.1
    }
    model = createModel4(params)
end

local loopConfig = {
    numEpoch = 10000,
    batchSize = 20,
    progressBar = false
}

local optimConfig = {
    learningRate = 0.1,
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
                    io.write(string.format('%d ', torch.round(data.testData[i][t][d])))
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
            if epoch % 5 == 0 then
                visualize()
                if opt.save then
                    logger:logInfo('saving model')
                    nnserializer.save(model, savePath)
                end
            end
        end
        )
end
