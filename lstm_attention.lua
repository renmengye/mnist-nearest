local synthqa = require('synthqa')
local optim_pkg = require('optim')
local utils = require('utils')
local lstm = require('lstm')
local nntrainer = require('nntrainer')
local logger = require('logger')()
local nnserializer = require('nnserializer')

function createSynthQAModel(params)
    -- params.objectEmbedDim
    -- params.colorEmbedDim
    -- params.questionLength
    -- params.wordEmbedDim
    -- params.lstmDim
    -- params.itemDim
    -- params.decoderSteps
    -- params.vocabSize
    -- params.numObject
    -- params.numColor
    -- params.attentionMechanism: 'soft' or 'hard'
    -- params.objective
    -- params.aggregatorWeights
    -- params.outputMapWeights
    
    if training == nil then
        training = false
    end

    -- Input
    local input = nn.Identity()()

    -- Items to attend
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
    local itemsJoinedReshape = mynn.BatchReshape(9, itemRawDim)(itemsJoined)
    itemsJoinedReshape.data.module.name = 'itemsJoinedReshape'

    -- Word Embeddings
    local wordIds = nn.Narrow(2, 55, params.questionLength)(input)
    local itemOfInterest = mynn.BatchReshape()(nn.Select(2, 3)(wordIds))
    local wordEmbed = nn.LookupTable(
        #synthqa.idict, params.wordEmbedDim)(
        nn.GradientStopper()(wordIds))
    local wordEmbedSeq = nn.SplitTable(2)(wordEmbed)
    local constantEncoderState = mynn.Constant(params.lstmDim * 2, 0)(input)

    -- Encoder LSTM
    local encoderCore = lstm.createUnit(
        params.wordEmbedDim, params.lstmDim)
    local encoder = nn.RNN(
        encoderCore, params.questionLength)(
        {wordEmbedSeq, constantEncoderState})
    local encoderStateSel = nn.SelectTable(
        params.questionLength)(encoder)
    encoder.data.module.name = 'encoder'
    local constantAttentionState = mynn.Constant(9)(input)
    local decoderStateInit = nn.JoinTable(2)(
        {encoderStateSel, constantAttentionState})
    decoderStateInit.data.module.name = 'decoderStateInit'

    -- Decoder dummy input
    local decoderInputConst = mynn.Constant(
        {params.decoderSteps, 1}, 0)(input)
    local decoderInputSplit = nn.SplitTable(2)(decoderInputConst)

    -- Decoder LSTM (1st layer)
    local decoderCore = lstm.createAttentionUnitDebug(
        1, params.lstmDim, 9, itemRawDim, 0.1)
    local decoder = nn.RNN(
        decoderCore, params.decoderSteps)(
        {decoderInputSplit, decoderStateInit, itemsJoinedReshape})
    decoder.data.module.name = 'decoder'
    local decoderJoin = mynn.BatchReshape(
        params.decoderSteps, params.lstmDim * 2 + 9)(nn.JoinTable(2)(decoder))
    local attention = nn.Narrow(3, params.lstmDim * 2 + 1, 9)(decoderJoin)

    -- Build entire model
    -- Need MSECriterion for regression reward.
    local all = nn.LazyGModule({input}, {attention})

    all:addModule('catEmbed', catEmbed)
    all:addModule('colorEmbed', colorEmbed)
    all:addModule('wordEmbed', wordEmbed)
    all:addModule('encoder', encoder)
    all:addModule('decoder', decoder)
    all:setup()

    -- Expand LSTMs
    encoder.data.module:expand()
    decoder.data.module:expand()
    all.criterion = nn.MSECriterion()
    all.decision = function(pred)
        return torch.round(pred * 100) / 100
    end
    return all
end

local N = 10000
local H = 10
local T = 8
local rawData = synthqa.genHowManyObject(N)
local data, _ = synthqa.prep(rawData, 'regression')
local labels = torch.Tensor(N, 8, 9)

local params = {
    objectEmbedDim = 2,
    colorEmbedDim = 2,
    questionLength = 3,
    wordEmbedDim = 10,
    lstmDim = H,
    itemDim = 10,
    decoderSteps = 8,
    vocabSize = #synthqa.idict,
    numObject = #synthqa.OBJECT + 1,
    numColor = #synthqa.COLOR + 1
}

for n = 1, #rawData do
    local objInterest = data[n][57]
    local sum = 0
    for i = 1, 9 do
        -- print(data[])
        if data[n][(i - 1) * 6 + 1] == objInterest then
            labels[{n, {}, i}] = 1
            sum = sum + 1
        end
    end
    if labels[n]:sum() > 0 then
        labels[n] = torch.round(labels[n] / sum * 100) / 100
    end
end
data = {
    trainData = data[{{1, torch.floor(N / 2)}}],
    trainLabels = labels[{{1, torch.floor(N / 2)}}],
    testData = data[{{torch.floor(N / 2) + 1, N}}],
    testLabels = labels[{{torch.floor(N / 2) + 1, N}}]
}

local gradClipTable = {
    catEmbed = 0.1,
    colorEmbed = 0.1,
    wordEmbed = 0.1,
    encoder = 0.1,
    decoder = 0.1
}
local model = createSynthQAModel(params)

local optimConfig = {
    learningRate = 0.01,
    gradientClip = utils.gradientClip(gradClipTable, model.sliceLayer)
}

local loopConfig = {
    numEpoch = 1000,
    batchSize = 20,
    progressBar = false
}
local optimizer = optim.adam

local nntrainer = NNTrainer(model, loopConfig, optimizer, optimConfig)

local trainEval = NNEvaluator('train', model)
local testEval = NNEvaluator('test', model, {
        -- NNEvaluator.getClassAccuracyAnalyzer(evalModel.decision, classes, labelStart),
        -- NNEvaluator.getClassConfusionAnalyzer(evalModel.decision, classes, labelStart),
        NNEvaluator.getAccuracyAnalyzer(model.decision)
    })
local evalModel = model

local visualizeAttention = function()
    logger:logInfo('attention analyzer')
    local numVisualize = 10
    local numItems = synthqa.NUM_GRID
    local testSubset = data.testData[{{1, numVisualize}}]
    local testRawSubset
    evalModel:forward(testSubset)
    local decoder = evalModel.moduleMap['decoder']
    local aggregator = evalModel.moduleMap['aggregator']
    local offset = torch.floor(N / 2)

    for n = 1, numVisualize do
        local rawDataItem = rawData[offset + n]
        print(string.format('%d. Q: %s (%d) A: %s', 
            n, rawDataItem.question, testSubset[n][-1], rawDataItem.answer))
        for i = 1, numItems do
            io.write(string.format('%10d', testSubset[n][(i - 1) * 6 + 1]))
        end
        io.write('\n')
        for t = 1, params.decoderSteps do
            for i = 1, numItems do
                local softAttention = decoder.replicas[t].moduleMap['softAttention']
                io.write(string.format(' %3.1f', softAttention.output[n][i]))
                io.write(string.format('/%3.1f', data.testLabels[n][t][i]))
                if torch.round(softAttention.output[n][i] * 10) == 
                    torch.round(data.testLabels[n][t][i] * 10) then
                    io.write('^ ')
                else
                    io.write('  ')
                end
            end
            io.write('\n')
        end
    end
end

local savePath = 'lstm_attention.w.h5'

nntrainer:trainLoop(
    data.trainData, data.trainLabels,
    function(epoch)
        if epoch % 1 == 0 then
            trainEval:evaluate(data.trainData, data.trainLabels)
            testEval:evaluate(data.testData, data.testLabels)
        end
        if epoch % 1 == 0 then
            visualizeAttention()
            nnserializer.save(model, savePath)
        end
    end
    )

