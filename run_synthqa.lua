local synthqa = require('synthqa')
local logger = require('logger')()
local utils = require('utils')
local nntrainer = require('nntrainer')
local nnevaluator = require('nnevaluator')
local nnserializer = require('nnserializer')
local reinforce_container = require('reinforce_container')
local adam = require('adam')

-------------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Synthetic Counting Training')
cmd:text()
cmd:text('Options:')
cmd:option('-train', false, 'whether to train a new network')
cmd:option('-path', 'synthqa.w.h5', 'save network path')
cmd:option('-save', false, 'whether to save the trained network')
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
    itemDim = 10,
    decoderSteps = 8,
    vocabSize = #synthqa.idict,
    numObject = #synthqa.OBJECT + 1,
    numColor = #synthqa.COLOR + 1,
    attentionMechanism = opt.attention,
    objective = opt.objective,
    aggregatorWeights = aggregatorTrained:read('lstm'):all(),
    outputMapWeights = aggregatorTrained:read('linear'):all()
}

local trainModel = synthqa.createModel(params, true)

-- for k,v in pairs(trainModel.moduleMap) do
--     logger:logInfo(k)
--     logger:logInfo(trainModel.sliceLayer(trainModel.w, k):size())
-- end

-- print(trainModel.w:size())
-- print(params.outputMapWeights)
-- print(trainModel.w[{{3570, 3585}}])
-- print(unpack(trainModel.parameterMap['answer']))
-- print(table.tostring(trainModel.parameterMap))
-- print(trainModel.sliceLayer(trainModel.w, 'answer'):eq(params.outputMapWeights))
-- print(trainModel.sliceLayer(trainModel.w, 'aggregator'):eq(params.aggregatorWeights))

local evalModel = synthqa.createModel(params, true)

-- local learningRateDecay = 0.001
-- local learningRates = {
--     catEmbed = 0.0001, 
--     colorEmbed = 0.0001,
--     wordEmbed = 0.0001,
--     encoder = 0.0001,
--     decoder = 0.0001,
--     aggregator = 0.00,
--     outputMap = 0.00
-- }
-- local learningRates = {
--     catEmbed = 0.01, 
--     colorEmbed = 0.01,
--     wordEmbed = 0.01,
--     encoder = 0.01,
--     decoder = 0.01,
--     aggregator = 0.00,
--     outputMap = 0.00
-- }

local learningRates = {
    catEmbed = 0.001, 
    colorEmbed = 0.001,
    wordEmbed = 0.001,
    encoder = 0.001,
    decoder = 0.001,
    aggregator = 0.00,
    outputMap = 0.00
}

local gradClipTable = {
    catEmbed = 1.0,
    colorEmbed = 1.0,
    wordEmbed = 1.0,
    encoder = 1.0,
    decoder = 1.0,
    aggregator = 1.0,
    outputMap = 1.0
}

-- local gradClipTable = {
--     catEmbed = 0.1,
--     colorEmbed = 0.1,
--     wordEmbed = 0.1,
--     encoder = 0.1,
--     decoder = 0.1,
--     aggregator = 0.1,
--     outputMap = 0.1
-- }

if params.attentionMechanism == 'hard' then
    -- learningRates['expectedReward'] = 0.01
    learningRates['expectedAttentionReward'] = 1.0
    learningRates['expectedCountingReward'] = 1.0
    gradClipTable['expectedAttentionReward'] = 1.0
    gradClipTable['expectedCountingReward'] = 1.0
    -- gradClipTable['expectedReward'] = 0.1
end

local optimConfig = {
    learningRate = 1.0,
    learningRates = utils.fillVector(
        torch.Tensor(trainModel.w:size()), trainModel.sliceLayer, learningRates),
    momentum = 0.9,
    gradientClip = utils.gradientClip(gradClipTable, trainModel.sliceLayer)
}

-- -- For Adam
-- local optimConfig = {
--     learningRate = 0.001,
--     learningRates = utils.fillVector(
--         torch.Tensor(trainModel.w:size()), trainModel.sliceLayer, learningRates),
--     gradientClip = utils.gradientClip(gradClipTable, trainModel.sliceLayer)
-- }

-- logger:logFatal(optimConfig.learningRates)

local loopConfig = {
    numEpoch = 10000,
    batchSize = 500,
    progressBar = true,
    analyzers = {classAccuracyAnalyzer},
}

-- local optimizer = optim.sgd
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
    local aggregator = evalModel.moduleMap['aggregator']
    local offset = torch.floor(N / 2)

    for n = 1, numVisualize do
        local rawDataItem = rawData[offset + n]
        local output = evalModel.decision(evalModel.moduleMap['outputMap'].output)[n][1]
        print(string.format('%d. Q: %s (%d) A: %s O: %s', 
            n, rawDataItem.question, testSubset[n][-1], rawDataItem.answer, 
            answerDict[output - labelStart + 1], output))
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
            local aggregatorInput = decoder.replicas[t].moduleMap['binaryOutput'].output
            local aggregatorInputSoft = decoder.replicas[t].moduleMap['binarySoftOutput'].output
            io.write(string.format('%6d', aggregatorInput[n][1]))
            io.write(string.format(' (%.2f)', aggregatorInputSoft[n][1]))
            io.write('\n')
        end
    end
end

local testEval = NNEvaluator('test', evalModel, 
    {
        NNEvaluator.getClassAccuracyAnalyzer(evalModel.decision, answerDict, labelStart),
        NNEvaluator.getClassConfusionAnalyzer(evalModel.decision, answerDict, labelStart),
        NNEvaluator.getAccuracyAnalyzer(evalModel.decision)
    })

-- visualizeAttention()
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
                if epoch % 20 == 0 then
                    logger:logInfo('saving model')
                    nnserializer.save(evalModel, opt.path)
                end
            end
        end
        )
end
