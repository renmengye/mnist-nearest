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
local statatt_model = require('synthqa_statatt_model')
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
cmd:option('-load', false, 'whether to load the trained network')
cmd:option('-loadpath', 'synthqa.w.h5', 'load network path')
cmd:option('-save', false, 'whether to save the trained network')
cmd:option('-savepath', 'synthqa.w.h5', 'save network path')
cmd:option('-num_ex', 10000, 'number of generated examples')
cmd:option('-attention', 'hard', 'hard or soft attention')
cmd:option('-model', 'super', 'super/unsuper/conv')
cmd:option('-viz', false, 'whether to visualize attention')
cmd:option('-loadpre', false, 'pretrained static attention network')
cmd:text()
opt = cmd:parse(arg)

logger:logInfo('--- command line options ---')
for key, value in pairs(opt) do
    logger:logInfo(string.format('%s: %s', key, value))
end
logger:logInfo('----------------------------')

-------------------------------------------------------------------------------
local dataset = synthqa.genHowManyObject(opt.num_ex)
local data, labels = synthqa.prep(dataset, 'regression')
data = {
    trainData = data[{{1, torch.floor(opt.num_ex / 2)}}],
    trainLabels = labels[{{1, torch.floor(opt.num_ex / 2)}}],
    testData = data[
        {{torch.floor(opt.num_ex / 2) + 1, opt.num_ex}}],
    testLabels = labels[
        {{torch.floor(opt.num_ex / 2) + 1, opt.num_ex}}]
}

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
    inputItemDim = 7, -- Encode item grid id as the 7th element
    itemDim = 8,
    numItems = dataset.maxNumItems,
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
    weightDecay = 0.000005
    -- gradientClip = utils.gradientClip(1.0)
}
params.answerDict = synthqa.NUMBER
params.labelStart = 0

logger:logInfo('---------- params ----------')
logger:logInfo(table.tostring(params))
logger:logInfo('----------------------------')

-------------------------------------------------------------------------------
if opt.loadpre then
    local pretrained = hdf5.open('synthqa_conv_model.w.h5', 'r')
    params.model.itemFilterLTWeights = pretrained:read('itemFilterLT'):all()
    params.model.encoderWeights = pretrained:read('encoder'):all()
    params.model.inputProcWeights = pretrained:read('inputProc'):all()
    modelLib.learningRates.inputProc = 0.0
    modelLib.learningRates.encoder = 0.0
    modelLib.learningRates.decoder = 0.0
end

-------------------------------------------------------------------------------
local model = {}
local modelLib
if opt.model == 'super' then
    modelLib = attspv_model
elseif opt.model == 'unsuper' then
    modelLib = attunspv_model
elseif opt.model == 'conv' then
    modelLib = conv_model
elseif opt.model == 'static' then
    modelLib = statatt_model
else
    logger:logFatal(string.format('Unknown model %s', opt.model))
end
model.train = modelLib.create(params.model, true)
model.eval = modelLib.create(params.model, true)
model.gradientClip = modelLib.gradientClip
model.learningRates = modelLib.learningRates

-------------------------------------------------------------------------------
if opt.load then
    nnserializer.load(model.train, opt.loadpath)
    model.eval.w:copy(model.train.w)
end
local optimizer = optim.adam2

-- Layer-wise learning rates or gradient clipping
if model.gradientClip then
    params.optimConfig.gradientClip = 
        utils.gradientClip(model.gradientClip, model.train.sliceLayer)
end
if model.learningRates then
    logger:logInfo('fill learning rates by layer')
    params.optimConfig.learningRates = 
        utils.fillVector(torch.Tensor(model.train.w:size()):zero(), 
            model.train.sliceLayer, model.learningRates)
    params.optimConfig.learningRate = 1.0
end

-------------------------------------------------------------------------------
local trainer = NNTrainer(
    model.train, params.loopConfig, optimizer, params.optimConfig)

-------------------------------------------------------------------------------
if not modelLib.getVisualize then
    modelLib.getVisualize = function(
        model, 
        softNormAttentionModule, 
        softAttentionSelModule, 
        hardAttentionModule, 
        recallerModule,
        data,
        dataset)
        return function()
            logger:logInfo('attention visualization')
            local numItems = model.params.numItems
            local outputTable = model:forward(data)
            for n = 1, data:size(1) do
                local rawDataItem = dataset[n]
                local output = outputTable[1][n][1]
                local realNumItems = 0
                print(string.format('%d. Q: %s (%d) A: %s O: %d', 
                    n, rawDataItem.question, data[n][-1], 
                    rawDataItem.answer, torch.round(output)))
                local itemsort, sortidx = data[n]:narrow(
                    1, 1, numItems * model.params.inputItemDim):reshape(
                    numItems, model.params.inputItemDim):select(
                    2, model.params.inputItemDim):sort()
                for i = 1, numItems do
                    local idx = sortidx[i]
                    local cat = data[n][
                        (idx - 1) * model.params.inputItemDim + 1]
                    local grid = data[n][idx * model.params.inputItemDim]
                    if cat == 4 and grid > synthqa.NUM_GRID then
                        break
                    end
                    io.write(string.format('%3d(%1d)', cat, grid))
                    realNumItems = realNumItems + 1
                end
                io.write('\n')
                for t = 1, model.params.decoderSteps do
                    -- local selIdx = 0
                    for i = 1, realNumItems do
                        local idx = sortidx[i]
                        if hardAttentionModule.output[n][t][idx] == 1.0 then
                            selIdx = idx
                            io.write(string.format(
                                ' %3.2f^', 
                                softNormAttentionModule.output[n][t][idx]))
                        else
                            io.write(string.format(
                                ' %3.2f ', 
                                softNormAttentionModule.output[n][t][idx]))
                        end
                    end
                    io.write(string.format(' (%.2f)', 
                        recallerModule.output[n][t][1]))
                    io.write(string.format(' (%.2f)', 
                        softAttentionSelModule.output[n][t][1]))
                    io.write('\n')
                end
            end
        end
    end
end

-------------------------------------------------------------------------------
local visualize
if opt.viz then
    local rawDataSubset = {}
    local numVisualize = 10
    local offset = torch.floor(params.data.numExamples / 2)
    for i = 1, numVisualize do
        table.insert(rawDataSubset, dataset[offset + i])
    end
    local testData = data.testData[{{1, numVisualize}}]
    if opt.model == 'conv' or opt.model == 'static' then
        visualize = modelLib.getVisualize(
            model.eval,
            model.eval.moduleMap['attentionReshape'],
            testData,
            rawDataSubset)
    else 
        visualize = modelLib.getVisualize(
            model.eval,
            model.eval.moduleMap['penAttentionValueReshape'],
            model.eval.moduleMap['softAttentionSelJoin'],
            model.eval.moduleMap['hardAttentionValueReshape'],
            model.eval.moduleMap['recallerBinaryReshape'],
            testData,
            rawDataSubset)
    end
    visualize()
end

-------------------------------------------------------------------------------
local trainEval = NNEvaluator('train', model.eval)
local testEval = NNEvaluator('test', model.eval, 
    {
        NNEvaluator.getClassAccuracyAnalyzer(
            model.eval.decision, params.answerDict, params.labelStart),
        NNEvaluator.getClassConfusionAnalyzer(
            model.eval.decision, params.answerDict, params.labelStart),
        NNEvaluator.getAccuracyAnalyzer(model.eval.decision)
    })

-------------------------------------------------------------------------------
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
                    nnserializer.save(model.eval, opt.savepath)
                end
            end
        end)
end
