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

function createSynthQAModel(params)
    -- Input
    local input = nn.Identity()()

    local itemRawDim = params.objectEmbedDim + params.colorEmbedDim + 4
    local items = nn.Narrow(2, 1, 54)(input)
    local itemsReshape = nn.Reshape(9, 6)(items)
    local catId = nn.Select(3, 1)(itemsReshape)
    local catIdReshape = mynn.BatchReshape()(catId)
    catIdReshape.data.module.name = 'catIdReshape'
    local colorId = nn.Select(3, 2)(itemsReshape)
    local colorIdReshape = mynn.BatchReshape()(colorId)
    colorIdReshape.data.module.name = 'colorIdReshape'
    local coord = nn.Narrow(3, 3, 4)(itemsReshape)
    local coordReshape = mynn.BatchReshape(4)(coord)
    coordReshape.data.module.name = 'coordReshape'
    local catEmbed = nn.LookupTable(
        params.numObject, params.objectEmbedDim)(
        nn.GradientStopper()(catIdReshape))
    local colorEmbed = nn.LookupTable(
        params.numColor, params.colorEmbedDim)(
        nn.GradientStopper()(colorIdReshape))
    local itemsJoined = nn.JoinTable(2, 2)(
        {catEmbed, colorEmbed, coordReshape})
    itemsJoined.data.module.name = 'itemsJoined'
    local itemsReshape = mynn.BatchReshape(9, itemRawDim)(itemsJoined)
    -- local itemsSplit = nn.SplitTable(2)(itemsJoined)

    -- Word Embeddings
    local wordIds = nn.Narrow(2, 55, params.questionLength)(input)
    local wordEmbed = nn.LookupTable(
        #synthqa.idict, params.wordEmbedDim)(
        nn.GradientStopper()(wordIds))

    local inputToLstm = nn.JoinTable(2)({wordEmbed, itemsReshape})
    inputToLstm.data.module.name = 'inputToLstm'
    local inputSplit = nn.SplitTable(2)(inputToLstm)

    local constState = mynn.Constant({2 * params.lstmDim}, 0)(input)
    local lstmCore = lstm.createUnit(params.wordEmbedDim, params.lstmDim)
    local lstm = nn.RNN(lstmCore, params.timespan)({inputSplit, constState})
    local lstmOutputSelTable = nn.SelectTable(params.timespan)(lstm)
    local lstmOutputSel = nn.Narrow(
        2, params.lstmDim + 1, params.lstmDim)(lstmOutputSelTable)
    local linear = nn.Linear(params.lstmDim, 1)(lstmOutputSel)

    local all = nn.LazyGModule({input}, {linear})
    all.criterion = nn.MSECriterion()
    all.decision = function (pred)
        return torch.round(pred):float()
    end

    all:addModule('catEmbed', catEmbed)
    all:addModule('colorEmbed', colorEmbed)
    all:addModule('wordEmbed', wordEmbed)
    all:addModule('lstm', lstm)
    all:addModule('linear', linear)
    all:setup()
    lstm.data.module:expand()
    return all
end

-------------------------------------------------------------------------------
function createModel(numHidden, timespan)
    local numInput = 1
    local modelInput = nn.Identity()()
    local splitTable = nn.SplitTable(2)(modelInput)
    local constState = mynn.Constant({2 * numHidden}, 0)(modelInput)
    local lstmCore = lstm.createUnit(numInput, numHidden)
    local lstm = nn.RNN(lstmCore, timespan)({splitTable, constState})
    -- local lstmOutputTableSel = nn.SelectTable(timespan)(lstm)
    local join = nn.JoinTable(2)(lstm)
    local lstmOutputReshape = mynn.BatchReshape(2 * numHidden)(join)
    local lstmOutputSel = nn.Narrow(
        2, numHidden + 1, numHidden)(lstmOutputReshape)
    local linear = nn.Linear(numHidden, 1)(lstmOutputSel)
    local sigmoid
    if opt.task == 'parity' then
        sigmoid = nn.Sigmoid()(linear)
    else
        sigmoid = linear
    end
    -- local outSplit = nn.SplitTable(2)(sigmoid)

    local all
    if opt.task == 'parity' then
        all = nn.LazyGModule({modelInput}, {sigmoid})
        all.criterion = nn.BCECriterion()
        all.decision = function (pred)
            return torch.gt(pred, 0.5):float()
        end
    else
        all = nn.LazyGModule({modelInput}, {sigmoid})
        all.criterion = nn.MSECriterion()
        all.decision = function (pred)
            return torch.round(pred):float()
        end
    end
    all:addModule('lstm', lstm)
    all:addModule('linear', linear)
    all:setup()
    lstm.data.module:expand()
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
        -- else if opt.task == 'parity' then
        --     local count = 0
        --     local attention = torch.floor(torch.uniform() * 2)
        --     for t = 2, T do
        --         data[i][t][1] = torch.floor(torch.uniform() * 2)
        --         count = count + data[i][t][1]
        --         label[i][t] = count % 2
        --     end
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
local N = 10000
local H = 10
local T = 20
local params

local data, labels, rawData, model, labelStart, classes
if opt.task == 'sum' or opt.task == 'parity' then
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
elseif opt.task == 'synthqa' then
    local rawData = synthqa.genHowManyObject(N)
    data, labels = synthqa.prep(rawData, 'regression')
    T = 12
    params = {
        objectEmbedDim = 2,
        colorEmbedDim = 2,
        wordEmbedDim = 8,
        lstmDim = H,
        itemDim = 10,
        timespan = T,
        questionLength = 3,
        vocabSize = #synthqa.idict,
        numObject = #synthqa.OBJECT + 1,
        numColor = #synthqa.COLOR + 1
    }
    data = {
        trainData = data[{{1, torch.floor(N / 2)}}],
        trainLabels = labels[{{1, torch.floor(N / 2)}}],
        testData = data[{{torch.floor(N / 2) + 1, N}}],
        testLabels = labels[{{torch.floor(N / 2) + 1, N}}]
    }
    labelStart = 0
    model = createSynthQAModel(params)
    evalModel = createSynthQAModel(params)
    classes = synthqa.NUMBER
end
local gradClipTable = {
    lstm = 0.1,
    linear = 0.1
}
local learningRates = {
    lstm = 0.8,
    linear = 0.8
}

local loopConfig = {
    numEpoch = 1000,
    batchSize = 20,
    progressBar = false
}

local optimizer = optim.sgd

local optimConfig = {
    learningRate = 0.1,
    learningRates = utils.fillVector(
        torch.Tensor(model.w:size()), model.sliceLayer, learningRates),
    momentum = 0.9,
    gradientClip = utils.gradientClip(gradClipTable, model.sliceLayer)
}

local savePath = 'lstm_parity.w.h5'
local nntrainer = NNTrainer(model, loopConfig, optimizer, optimConfig)
local trainEval = NNEvaluator('train', model)
local testEval = NNEvaluator('test', model, {
        -- NNEvaluator.getClassAccuracyAnalyzer(evalModel.decision, classes, labelStart),
        -- NNEvaluator.getClassConfusionAnalyzer(evalModel.decision, classes, labelStart),
        NNEvaluator.getAccuracyAnalyzer(evalModel.decision)
    })

local visualizeSeq = function()
    local numItems = 1
    local T = 25

    local evalModel = createModel(H, T)
    local testSubset
    if opt.task == 'parity' or opt.task == 'sum' then
        testSubset = getData(numItems * 2, T).testData
        evalModel.w:copy(model.w)
        evalModel:forward(testSubset)
        for n = 1, numItems do
            print('example', n)
            for t = 1, T do
                io.write(string.format('%6d', testSubset[n][t][1]))
                local hidden = evalModel.moduleMap['lstm'].replicas[t].output[n][{{H + 1, H + H}}]
                for d = 1, H do
                    io.write(string.format('%6.2f', hidden[d]))
                end
                local linear = evalModel.moduleMap['linear']
                local y = linear:forward(hidden)

                if opt.task == 'parity' then
                    y = 1 / (1 + (torch.exp(-y[1])))
                else
                    y = y[1]
                end
                io.write(string.format('%6.2f', y))
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
        if epoch % 10 == 0 then
            trainEval:evaluate(data.trainData, data.trainLabels)
            testEval:evaluate(data.testData, data.testLabels)
            visualizeSeq()
        end
        if epoch % 100 == 0 then
            logger:logInfo('saving model')
            nnserializer.save(model, savePath)
        end
    end
    )

if opt.save then
    nnserializer.save(model, opt.path)
end
