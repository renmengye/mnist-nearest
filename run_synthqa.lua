local synthqa = require('synthqa')
local logger = require('logger')()
local utils = require('utils')
local nntrainer = require('nntrainer')
local nnevaluator = require('nnevaluator')
local nnserializer = require('nnserializer')
local adam = require('adam')
local attspv_model = require('synthqa_attspv_model')
local attunspv_model = require('synthqa_attunspv_model')
local conv_model = require('synthqa_conv_model')
local optim_pkg = require('optim')
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
cmd:option('-model', 'super', 'super/unsuper/conv')
cmd:option('-viz', false, 'whether to visualize attention')
-- cmd:option('-reinforce', true, 'whether has expected reward')
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
local modelLib
if opt.model == 'super' then
    modelLib = attspv_model
elseif opt.model == 'unsuper' then
    modelLib = attunspv_model
elseif opt.model == 'conv' then
    modelLib = conv_model
else
    logger:logFatal(string.format('Unknown model %s', opt.model))
end
model.train = modelLib.create(params.model, true)
model.eval = modelLib.create(params.model, true)
model.gradClipTable = modelLib.gradClipTable
model.learningRates = modelLib.learningRates

model.train.sliceLayer(model.train.w, 'catEmbed'):copy(
    torch.Tensor({{0, 0}, {0, 1}, {1, 0}, {1, 1}}))
if opt.load then
    nnserializer.load(model.train, opt.path)
end
local optimizer = optim.adam2
if model.gradientClip then
    params.optimConfig.gradientClip = 
        utils.gradientClip(model.gradClipTable, model.train.sliceLayer)
end
if model.learningRates then
    params.optimConfig.learningRates = 
        utils.fillVector(torch.Tensor(model.train.w:size()):zero(), 
            model.train.sliceLayer, model.learningRates)
end
local trainer = NNTrainer(
    model.train, params.loopConfig, optimizer, params.optimConfig)
if not modelLib.getVisualize then
    modelLib.getVisualize = function(
        model, 
        softNormAttentionModule, 
        softUnnormAttentionModule, 
        hardAttentionModule, 
        recallerModule,
        data,
        rawData)
        return function()
            logger:logInfo('attention visualization')
            local numItems = synthqa.NUM_GRID
            local outputTable = model:forward(data)
            for n = 1, data:size(1) do
                local rawDataItem = rawData[n]
                local output = outputTable[1][n][1]
                print(string.format('%d. Q: %s (%d) A: %s O: %d', 
                    n, rawDataItem.question, data[n][-1], 
                    rawDataItem.answer, output))
                for i = 1, numItems do
                    io.write(string.format('%6d', data[n][(i - 1) * 6 + 1]))
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
end

local visualize
if opt.viz then
    local rawDataSubset = {}
    local numVisualize = 10
    local offset = torch.floor(params.data.numExamples / 2)
    for i = 1, numVisualize do
        table.insert(rawDataSubset, rawData[offset + i])
    end
    local testData = data.testData[{{1, numVisualize}}]
    if opt.model == 'conv' then
        visualize = modelLib.getVisualize(
            model.eval,
            model.eval.moduleMap['attentionReshape'],
            testData,
            rawDataSubset)
    else 
        visualize = modelLib.getVisualize(
            model.eval,
            model.eval.moduleMap['penAttentionValueReshape'],
            model.eval.moduleMap['softAttentionValueReshape'],
            model.eval.moduleMap['hardAttentionValueReshape'],
            model.eval.moduleMap['recallerBinaryReshape'],
            testData,
            rawDataSubset)
    end
    visualize()
end
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
            if epoch % 5 == 0 and opt.viz then
                visualize()
            end
            if opt.save then
                if epoch % 1 == 0 then
                    logger:logInfo('saving model')
                    nnserializer.save(model.eval, opt.path)
                end
            end
        end)
end
