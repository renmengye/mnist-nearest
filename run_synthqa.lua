local synthqa = require('synthqa')
local logger = require('logger')()
local utils = require('utils')
local nntrainer = require('nntrainer')
local nnevaluator = require('nnevaluator')
local nnserializer = require('nnserializer')
local reinforce_container = require('reinforce_container')

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
cmd:option('-attention', 'soft', 'soft or hard attention')
cmd:option('-objective', 'classification', 'classification or regression')
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
logger:logInfo(data:size(), 1)
logger:logInfo(labels:size(), 1)
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
local learningRates = {
    catEmbed = 0.01, 
    colorEmbed = 0.01,
    wordEmbed = 0.01,
    encoder = 0.01,
    decoder = 0.01,
    binaryInput = 0.01,
    aggregator = 0.00,
    answer = 0.00
}

local gradClipTable = {
    catEmbed = 0.1,
    colorEmbed = 0.1,
    wordEmbed = 0.1,
    encoder = 0.1,
    decoder = 0.1,
    aggregator = 0.1,
    answer = 0.1
}

if attentionMechanism == 'hard' then
    -- learningRates['expectedReward'] = 0.01
    learningRates['expectedReward'] = 0.1
    gradClipTable['expectedReward'] = 0.1
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
--     learningRate = 0.001
-- }

local loopConfig = {
    numEpoch = 10000,
    batchSize = 200,
    progressBar = true,
    analyzers = {classAccuracyAnalyzer},
}

local optimizer = optim.sgd
-- local optimizer = optim.adam
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
    -- local evalModel = trainModel
    logger:logInfo('attention analyzer')
    local numVisualize = 10
    local numItems = synthqa.NUM_GRID
    local testSubset = data.testData[{{1, numVisualize}}]
    local testRawSubset
    evalModel:forward(testSubset)
    local decoder = evalModel.moduleMap['decoder']
    local aggregator = evalModel.moduleMap['aggregator']
    local binaryInput = evalModel.moduleMap['binaryInput']
    local offset = torch.floor(N / 2)
    for n = 1, numVisualize do
        local rawDataItem = rawData[offset + n]
        local output = evalModel.decision(evalModel.moduleMap['answer'].output)[n][1]
        print(string.format('%d. Q: %s (%d) A: %s O: %s, %s', 
            n, rawDataItem.question, testSubset[n][-1], rawDataItem.answer, 
            answerDict[output - labelStart + 1], output))
        for i = 1, numItems do
            io.write(string.format('%5d', rawDataItem.items[i].category))
        end
        io.write('\n')
        for t = 1, params.decoderSteps do
            for i = 1, numItems do
                local attention = decoder.replicas[t].moduleMap['attention']
                io.write(string.format('%5.2f', attention.output[n][i]))
            end
            local aggregatorInput = binaryInput.replicas[t].moduleMap['input'].output
            io.write(string.format('%5.2f', 
                aggregatorInput[n][1]))
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
                trainEval:evaluate(data.trainData, data.trainLabels)
            end
            if epoch % 3 == 0 then
                testEval:evaluate(data.testData, data.testLabels)
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
