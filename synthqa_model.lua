local synthqa = require('synthqa')

local logger = require('logger')()
local utils = require('utils')
local nntrainer = require('nntrainer')
local nnevaluator = require('nnevaluator')
local nnserializer = require('nnserializer')
local reinforce_container = require('reinforce_container')
local adam = require('adam')
local lstm = require('lstm')
local vr_attention_reward = require('vr_attention_reward')
local counting_criterion = require('counting_criterion')
local double_counting_criterion = require('double_counting_criterion')
local attention_criterion = require('attention_criterion')
-- local relatedness_criterion = require('relatedness_criterion')

-------------------------------------------------------------------------------
function synthqa.createModel2(params, training)
    -- params.objectEmbedDim
    -- params.colorEmbedDim
    -- params.questionLength
    -- params.wordEmbedDim
    -- params.lstmDim
    -- params.aggregatorDim
    -- params.itemDim
    -- params.numItems
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
    local items = nn.Narrow(2, 1, params.numItems * 6)(input)
    local itemsReshape = nn.Reshape(params.numItems, 6)(items)
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
        mynn.GradientStopper()(catIdReshape))
    local colorEmbed = nn.LookupTable(
        params.numColor, params.colorEmbedDim)(
        mynn.GradientStopper()(colorIdReshape))
    local itemsJoined = nn.JoinTable(2, 2)(
        {catEmbed, colorEmbed, coordReshape})
    itemsJoined.data.module.name = 'itemsJoined'
    local itemsJoinedReshape = mynn.BatchReshape(
        params.numItems, params.itemDim)(itemsJoined)
    itemsJoinedReshape.data.module.name = 'itemsJoinedReshape'

    -- Word Embeddings
    local wordIds = nn.Narrow(
        2, params.numItems * 6 + 1, params.questionLength)(input)
    local itemOfInterest = nn.Select(2, params.numItems * 6 + 3)(input)
    local wordEmbed = nn.LookupTable(
        #synthqa.idict, params.wordEmbedDim)(
        mynn.GradientStopper()(wordIds))
    local wordEmbedSeq = nn.SplitTable(2)(wordEmbed)

    -- Encoder LSTM
    local constantEncoderState = mynn.Constant(params.lstmDim * 2, 0)(input)
    local encoderCore = lstm.createUnit(
        params.wordEmbedDim, params.lstmDim)
    local encoder = nn.RNN(
        encoderCore, params.questionLength)(
        {wordEmbedSeq, constantEncoderState})
    local encoderStateSel = nn.SelectTable(
        params.questionLength)(encoder)
    encoder.data.module.name = 'encoder'
    local constantAttendedState = mynn.Constant(
        params.itemDim + params.numItems * 2)(input)

    -- Decoder LSTM (Attend to the object of interest)
    local decoderStateInit = nn.JoinTable(2)(
        {encoderStateSel, constantAttendedState})
    decoderStateInit.data.module.name = 'decoderStateInit'
    local decoderInputConst = mynn.Constant(
        {params.decoderSteps, 1}, 0)(input)
    local decoderInputSplit = nn.SplitTable(2)(decoderInputConst)

    local decoderCore = lstm.createAttentionUnit(
        1, params.lstmDim, params.numItems, params.itemDim, 0.1, 
        params.attentionMechanism, training)
    local decoder = nn.RNN(
        decoderCore, params.decoderSteps)(
        {decoderInputSplit, decoderStateInit, itemsJoinedReshape})
    decoder.data.module.name = 'decoder'

    local attentionOutputs = {}
    local attentionValues = {}
    local attentionSel = {}
    for t = 1, params.decoderSteps do
        table.insert(attentionOutputs, nn.Narrow(
            2, params.lstmDim * 2 + 1, params.itemDim)(
            nn.SelectTable(t)(decoder)))
        table.insert(attentionValues, nn.Narrow(
            2, params.lstmDim * 2 + params.itemDim + 1, params.numItems)(
            nn.SelectTable(t)(decoder)))
        table.insert(attentionSel, nn.Narrow(
            2, 
            params.lstmDim * 2 + params.itemDim + params.numItems * 2 + 1, 1)(
            nn.SelectTable(t)(decoder)))
    end
    local attentionValueJoin = nn.JoinTable(2)(attentionValues)
    attentionValueJoin.data.module.name = 'attentionValueJoin'
    local attentionValueReshape = mynn.BatchReshape(
        params.decoderSteps, params.numItems)(attentionValueJoin)
    local attentionOutputTable = nn.Identity()(attentionOutputs)
    local attentionSelJoin = mynn.BatchReshape(params.decoderSteps, 1)(
        nn.JoinTable(2)(attentionSel))

    -- Recaller LSTM (Tell whether you have seen the object before)
    local recallerStateInit = encoderStateSel
    local recaller = nn.RNN(
        lstm.createUnit(params.itemDim, params.lstmDim), params.decoderSteps)(
        {attentionOutputTable, recallerStateInit})
    recaller.data.module.name = 'recaller'

    local recallerOutputs = {}
    for t = 1, params.decoderSteps do
        table.insert(recallerOutputs, nn.Narrow(
            2, params.lstmDim + 1, params.lstmDim)(
            nn.SelectTable(t)(recaller)))
    end
    local recallerJoin = mynn.BatchReshape(params.lstmDim)(
        nn.JoinTable(2)(recallerOutputs))
    local recallerOutputMap = nn.Linear(params.lstmDim, 1)(recallerJoin)
    local recallerBinary = nn.Sigmoid()(recallerOutputMap)
    local recallerBinaryReshape = mynn.BatchReshape(params.decoderSteps, 1)(
        recallerBinary)

    local recallerAttMul = mynn.GradientStopper()(nn.CMulTable()(
        {recallerBinaryReshape, attentionSelJoin}))
    local recallerBinarySplit = nn.SplitTable(2)(recallerAttMul)

    -- Aggregator (adds 1's and 0's)
    local constantAggState = mynn.Constant(params.aggregatorDim * 2, 0)(input)
    local aggregator = nn.RNN(
        lstm.createUnit(1, params.aggregatorDim), params.decoderSteps)(
        {recallerBinarySplit, constantAggState})
    
    -- Load pretrained weights
    if params.aggregatorWeights then
        local agg_w, agg_dl_dw = aggregator.data.module.core:getParameters()
        agg_w:copy(params.aggregatorWeights)
    end
    aggregator.data.module.name = 'aggregator'

    -- Classify answer
    local aggregatorOutputs = {}
    for t = 1, params.decoderSteps do
        table.insert(aggregatorOutputs, nn.Narrow(
            2, params.aggregatorDim + 1, params.aggregatorDim)(
            nn.SelectTable(t)(aggregator)))
    end
    local aggregatorOutputJoin = nn.JoinTable(2)(aggregatorOutputs)
    aggregatorOutputJoin.data.module.name = 'aggregatorOutputJoin'
    local aggregatorOutputReshape = mynn.BatchReshape(
        params.aggregatorDim)(aggregatorOutputJoin)
    local outputMap = nn.Linear(params.aggregatorDim, 1)(
        aggregatorOutputReshape)
    local outputMapReshape = mynn.BatchReshape(
        params.decoderSteps, 1)(outputMap)

    -- Load pretrained weights
    if params.outputMapWeights then
        local ow, odldw = outputMap.data.module:getParameters()
        ow:copy(params.outputMapWeights)
    end
    local final = nn.Select(2, params.decoderSteps)(outputMapReshape)

    local attentionOut = nn.Identity()(
        {attentionValueReshape, itemOfInterest, catIdReshape})

    all = nn.LazyGModule({input}, 
        {final, outputMapReshape, attentionOut, 
        recallerBinaryReshape})

    all:addModule('catEmbed', catEmbed)
    all:addModule('colorEmbed', colorEmbed)
    all:addModule('wordEmbed', wordEmbed)
    all:addModule('encoder', encoder)
    all:addModule('decoder', decoder)
    all:addModule('recaller', recaller)
    all:addModule('recallerOutputMap', recallerOutputMap)
    all:addModule('recallerBinaryReshape', recallerBinaryReshape)
    all:addModule('attentionSelJoin', attentionSelJoin)
    all:addModule('aggregator', aggregator)
    all:addModule('outputMap', outputMap)
    all:addModule('final', final)
    all:setup()

    -- Expand LSTMs
    encoder.data.module:expand()
    decoder.data.module:expand()
    recaller.data.module:expand()
    aggregator.data.module:expand()

    if params.attentionMechanism == 'soft' then
        if params.objective == 'regression' then
            all.criterion = nn.MSECriterion()
        elseif params.objective == 'classification' then
            all.criterion = nn.ClassNLLCriterion()
        else
            logger:logFatal(string.format(
                'unknown training objective %s', params.objective))
        end
    elseif params.attentionMechanism == 'hard' then
        -- Setup criterions and rewards
        if params.objective == 'regression' then
            all.criterion = nn.ParallelCriterion(true)
              :add(nn.MSECriterion(), 0.1)
              :add(mynn.CountingCriterion(
                recallerAttMul.data.module), 0.1)
              :add(mynn.AttentionCriterion(decoder.data.module), 1.0)
              :add(mynn.DoubleCountingCriterion(decoder.data.module), 1.0)
        else
            logger:logFatal(string.format(
                'unknown training objective %s', params.objective))
        end
    end

    if params.objective == 'regression' then
        all.decision = function(pred)
            local num = torch.round(pred)
            return num
        end        
    else
        logger:logFatal(string.format(
            'unknown training objective %s', params.objective))
    end

    return all
end

-------------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Synthetic Counting Training')
cmd:text()
cmd:text('Options:')
cmd:option('-train', false, 'whether to train a new network')
cmd:option('-path', 'synthqa.w.h5', 'save network path')
cmd:option('-save', false, 'whether to save the trained network')
cmd:option('-load', false, 'whether to load the trained network')
cmd:option('-num_ex', 10000, 'number of generated examples')
cmd:option('-attention', 'hard', 'soft or hard attention')
cmd:option('-objective', 'regression', 'classification or regression')
cmd:text()
opt = cmd:parse(arg)

logger:logInfo('--- command line options ---')
for key, value in pairs(opt) do
    logger:logInfo(string.format('%s: %s', key, value))
end
logger:logInfo('----------------------------')

-------------------------------------------------------------------------------
local N = opt.num_ex
-- local rawData1 = synthqa.genHowManyObject(N / 2)
-- local rawData2 = synthqa.genHowManyObject(N / 2)
-- synthqa.checkOverlap(rawData1, rawData2)

local rawData = synthqa.genHowManyObject(N)
local data, labels = synthqa.prep(rawData, opt.objective)

data = {
    trainData = data[{{1, torch.floor(N / 2)}}],
    trainLabels = labels[{{1, torch.floor(N / 2)}}],
    testData = data[{{torch.floor(N / 2) + 1, N}}],
    testLabels = labels[{{torch.floor(N / 2) + 1, N}}]
}

local aggregatorTrained = hdf5.open('lstm_sum.w.h5')

local params = {
    objectEmbedDim = 2,
    colorEmbedDim = 2,
    questionLength = 3,
    wordEmbedDim = 10,
    lstmDim = 10,
    aggregatorDim = 10,
    itemDim = 8,
    numItems = 9,
    decoderSteps = 12,
    vocabSize = #synthqa.idict,
    numObject = #synthqa.OBJECT + 1,
    numColor = #synthqa.COLOR + 1,
    attentionMechanism = opt.attention,
    objective = opt.objective,
    -- aggregatorWeights = aggregatorTrained:read('lstm'):all(),
    -- outputMapWeights = aggregatorTrained:read('linear'):all()
}

local trainModel = synthqa.createModel2(params, true)
if opt.load then
    nnserializer.load(trainModel, opt.path)
end
local evalModel = synthqa.createModel2(params, true)

local gradClipTable = {
    catEmbed = 0.1,
    colorEmbed = 0.1,
    wordEmbed = 0.1,
    encoder = 0.1,
    decoder = 0.1,
    recaller = 0.1,
    recallerOutputMap = 0.1,
    aggregator = 0.1,
    outputMap = 0.1
}

local optimConfig = {
    learningRate = 0.001,
    momentum = 0.9,
    weightDecay = 0.000005,
    gradientClip = utils.gradientClip(gradClipTable, trainModel.sliceLayer)
}

local loopConfig = {
    numEpoch = 10000,
    batchSize = 64,
    progressBar = true,
    analyzers = {classAccuracyAnalyzer},
}

local optimizer = optim.adam2
local trainer = NNTrainer(trainModel, loopConfig, optimizer, optimConfig)
local trainEval = NNEvaluator('train', evalModel)

local answerDict, labelStart
if opt.objective == 'regression' then
    answerDict = synthqa.NUMBER
    labelStart = 0
else
    answerDict = synthqa.idict
    labelStart = 1
end

local visualizeAttention = function()
    logger:logInfo('attention analyzer')
    local numVisualize = 10
    local numItems = synthqa.NUM_GRID
    local testSubset = data.testData[{{1, numVisualize}}]
    local testRawSubset
    evalModel:forward(testSubset)
    local decoder = evalModel.moduleMap['decoder']
    local recallerBinaryReshape = evalModel.moduleMap['recallerBinaryReshape']
    local attentionSelReshape = evalModel.moduleMap['attentionSelJoin']
    local aggregator = evalModel.moduleMap['aggregator']
    local offset = torch.floor(N / 2)

    for n = 1, numVisualize do
        local rawDataItem = rawData[offset + n]
        local output = evalModel.decision(evalModel.moduleMap['final'].output)
        output = output[n][1]
        print(string.format('%d. Q: %s (%d) A: %s O: %s', 
            n, rawDataItem.question, testSubset[n][-1], rawDataItem.answer, output))
        for i = 1, numItems do
            io.write(string.format('%6d', testSubset[n][(i - 1) * 6 + 1]))
        end
        io.write('\n')
        for t = 1, params.decoderSteps do
            for i = 1, numItems do
                local attention = decoder.replicas[t].moduleMap['attention']
                local softAttention = decoder.replicas[t].moduleMap['softAttention']
                if attention.output[n][i] == 1.0 then
                    io.write(string.format(' %3.2f^', softAttention.output[n][i]))
                elseif attention.output[n][i] == 0.0 then
                    io.write(string.format(' %3.2f ', softAttention.output[n][i]))
                else
                    io.write(string.format('%6.2f', attention.output[n][i]))
                end
            end
            local aggregatorInputSoft = recallerBinaryReshape.output[{{}, t, {}}]
            io.write(string.format(' (%.2f)', aggregatorInputSoft[n][1]))
            local attentionScore = attentionSelReshape.output[{{}, t, {}}]
            io.write(string.format(' (%.2f)', attentionScore[n][1]))
            io.write('\n')
        end
    end
end

-- nnserializer.load(evalModel, 'debugnan.w.h5')
-- print(evalModel.sliceLayer(evalModel.w, 'encoder'))
-- print(evalModel.sliceLayer(evalModel.w, 'wordEmbed'))
-- print(evalModel.sliceLayer(evalModel.w, 'catEmbed'))
-- print(evalModel.sliceLayer(evalModel.w, 'colorEmbed'))
visualizeAttention()
-- for t = 1, 12 do
--     print(evalModel.moduleMap['decoder'].replicas[t].output)
-- end

-- logger:logFatal('error')

local testEval = NNEvaluator('test', evalModel, 
    {
        NNEvaluator.getClassAccuracyAnalyzer(evalModel.decision, answerDict, labelStart),
        NNEvaluator.getClassConfusionAnalyzer(evalModel.decision, answerDict, labelStart),
        NNEvaluator.getAccuracyAnalyzer(evalModel.decision)
    })
visualizeAttention()
if opt.train then
    trainer:trainLoop(
        data.trainData, data.trainLabels,
        function (epoch)
            -- Copy the weights from the training model
            evalModel.w:copy(trainModel.w)
            if epoch % 1 == 0 then
                trainEval:evaluate(data.trainData, data.trainLabels, 1000)
            end
            if epoch % 5 == 0 then
                testEval:evaluate(data.testData, data.testLabels, 1000)
            end
            if epoch % 5 == 0 then
                visualizeAttention()
            end
            if opt.save then
                if epoch % 1 == 0 then
                    logger:logInfo('saving model')
                    nnserializer.save(evalModel, opt.path)
                end
            end
        end)
end
