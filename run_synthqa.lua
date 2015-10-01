local synthqa = require('synthqa')
local logger = require('logger')()
local utils = require('utils')
local nntrainer = require('nntrainer')
local nnevaluator = require('nnevaluator')
local nnserializer = require('nnserializer')
local adam = require('adam')
local attspv_model = require('synthqa_attspv_model')
torch.setnumthreads(1)

-------------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Synthetic Counting Training')
cmd:text()
cmd:text('Options:')
cmd:option('-name', 'synthqa', 'name of the thing')
cmd:option('-train', false, 'whether to train a new network')
cmd:option('-save', false, 'whether to save the trained network')
cmd:option('-load', false, 'whether to load the trained network')
cmd:option('-path', 'synthqa.w.h5', 'save network path')
cmd:option('-num_ex', 10000, 'number of generated examples')
cmd:option('-attention', 'hard', 'hard or soft attention')
cmd:text()
opt = cmd:parse(arg)

logger:logInfo('--- command line options ---')
for key, value in pairs(opt) do
    logger:logInfo(string.format('%s: %s', key, value))
end
logger:logInfo('----------------------------')

-------------------------------------------------------------------------------
local params = {}
params.name = opt.name
params.data = {
    numExamples = opt.num_ex
}
params.model = {
    objectEmbedDim = 2,
    colorEmbedDim = 2,
    questionLength = 3,
    wordEmbedDim = 10,
    encoderDim = 10,
    recallerDim = 10,
    aggregatorDim = 10,
    itemDim = 8,
    numItems = 9,
    decoderSteps = 12,
    vocabSize = #synthqa.idict,
    numObject = #synthqa.OBJECT + 1,
    numColor = #synthqa.COLOR + 1,
    attentionMechanism = opt.attention,
    objective = opt.objective
}
params.gradClipTable = {
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
params.loopConfig = {
    numEpoch = 10000,
    batchSize = 64,
    progressBar = true,
}
params.optimConfig = {
    learningRate = 0.1,
    -- Decay to 0.1 after 100 epochs
    learningRateDecay = 0.1 / (opt.num_ex / 2 / params.loopConfig.batchSize),
    momentum = 0.9,
    weightDecay = 0.000005
}
params.answerDict = synthqa.NUMBER
params.labelStart = 0

logger:logInfo('---------- params ----------')
logger:logInfo(table.tostring(params))
logger:logInfo('----------------------------')
local rawData = synthqa.genHowManyObject(params.data.numExamples)
local data, labels = synthqa.prep(rawData, 'regression')
data = {
    trainData = data[{{1, torch.floor(params.data.numExamples / 2)}}],
    trainLabels = labels[{{1, torch.floor(params.data.numExamples / 2)}}],
    testData = data[
        {{torch.floor(params.data.numExamples / 2) + 1, 
        params.data.numExamples}}],
    testLabels = labels[
        {{torch.floor(params.data.numExamples / 2) + 1, 
        params.data.numExamples}}]
}
local model = {}
model.train = attspv_model.create(params.model, true)
model.eval = attspv_model.create(params.model, true)
model.train.sliceLayer(model.train.w, 'catEmbed'):copy(
    torch.Tensor({{0, 0}, {0, 1}, {1, 0}, {1, 1}}))
if opt.load then
    nnserializer.load(model.train, opt.path)
end
local optimizer = optim.adam2
params.optimConfig.gradientClip = 
    utils.gradientClip(params.gradClipTable, model.train.sliceLayer)
local trainer = NNTrainer(
    model.train, params.loopConfig, optimizer, params.optimConfig)
local getVisualizeAttention = function(
    evalModel, 
    softNormAttentionModule, 
    softUnnormAttentionModule, 
    hardAttentionModule, 
    recallerModule)
        return function()
        logger:logInfo('attention visualization')
        local numVisualize = 10
        local numItems = synthqa.NUM_GRID
        local testSubset = data.testData[{{1, numVisualize}}]
        local testRawSubset
        local outputTable = model.eval:forward(testSubset)
        local offset = torch.floor(params.data.numExamples / 2)
        for n = 1, numVisualize do
            local rawDataItem = rawData[offset + n]
            local output = outputTable[1][n][1]
            print(string.format('%d. Q: %s (%d) A: %s O: %d', 
                n, rawDataItem.question, testSubset[n][-1], 
                rawDataItem.answer, output))
            for i = 1, numItems do
                io.write(string.format('%6d', testSubset[n][(i - 1) * 6 + 1]))
            end
            io.write('\n')
            for t = 1, params.model.decoderSteps do
                local selIdx = 1
                for i = 1, numItems do
                    if hardAttentionModule.output[n][t][i] == 1.0 then
                        selIdx = i
                        io.write(string.format(
                            ' %3.2f^', 
                            softNormAttentionModule.output[n][t][i]))
                    else
                        io.write(string.format(
                            ' %3.2f ', 
                            softNormAttentionModule.output[n][t][i]))
                    end
                end
                io.write(string.format(' (%.2f)', 
                    recallerModule.output[n][t][1]))
                io.write(string.format(' (%.2f)', 
                    softUnnormAttentionModule.output[n][t][selIdx]))
                io.write('\n')
            end
        end
    end
end

local visualizeAttention = getVisualizeAttention(
    model.eval,
    model.eval.moduleMap['penAttentionValueReshape'],
    model.eval.moduleMap['softAttentionValueReshape'],
    model.eval.moduleMap['hardAttentionValueReshape'],
    model.eval.moduleMap['recallerBinaryReshape'])
visualizeAttention()
local trainEval = NNEvaluator('train', model.eval)
local testEval = NNEvaluator('test', model.eval, 
    {
        NNEvaluator.getClassAccuracyAnalyzer(
            model.eval.decision, params.answerDict, params.labelStart),
        NNEvaluator.getClassConfusionAnalyzer(
            model.eval.decision, params.answerDict, params.labelStart),
        NNEvaluator.getAccuracyAnalyzer(model.eval.decision)
    })
if opt.train then
    trainer:trainLoop(
        data.trainData, data.trainLabels,
        function (epoch)
            -- Copy the weights from the training model
            model.eval.w:copy(model.train.w)
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
                    nnserializer.save(model.eval, opt.path)
                end
            end
        end)
end
