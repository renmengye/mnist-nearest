local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local logger = require('logger')()
local synthseg = require('synthseg')

local batch_reshape = require('batch_reshape')
local lazy_gmodule = require('lazy_gmodule')

local optim_pkg = require('optim')
local adam2 = require('adam')
local nntrainer = require('nntrainer')
local nnevaluator = require('nnevaluator')

-------------------------------------------------------------------------------
function createModel(params)
    -- params.itemDim
    -- params.numItems
    -- params.queryDim
    local input = nn.Identity()()
    local itemsNarrow = nn.Narrow(
        2, 1, params.numItems * params.itemDim)(input)
    local queryNarrow = nn.Narrow(
        2, params.numItems * params.itemDim + 1, params.queryDim)(input)
    local queryLT = nn.Linear(params.queryDim, params.itemDim)(queryNarrow)
    local queryFilter = mynn.BatchReshape(
        params.itemDim, 1)(nn.Tanh()(queryLT))
    local itemsReshape = mynn.BatchReshape(
        params.numItems, params.itemDim)(itemsNarrow)

    local itemsReshape2 = mynn.BatchReshape(params.itemDim)(itemsNarrow)
    local itemsLT = nn.Linear(params.itemDim, params.itemDim)(itemsReshape2)
    local itemsLTReshape = mynn.BatchReshape(params.numItems, params.itemDim)(itemsLT)
    local itemsTanh = nn.Tanh()(itemsLTReshape)
    local attention = nn.MM()({itemsTanh, queryFilter})
    
    -- local attention = nn.MM()({itemsReshape, queryFilter})
    local final = nn.Sigmoid()(attention)
    local all = nn.LazyGModule({input}, {final})
    all:addModule('queryLT', queryLT)
    all:addModule('itemsLT', itemsLT)
    all:setup()
    all.criterion = nn.BCECriterion()
    all.decision = function(pred)
        local num = torch.round(pred)
        return num
    end
    return all
end

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

-------------------------------------------------------------------------------
local N = opt.num_ex
local dataset = synthseg.gen(N)
local items = synthseg.getItems(dataset)
local alldata = synthseg.encodeDataset(dataset)
local alllabels = synthseg.getSemanticLabels(dataset)
local data = {
    trainData = alldata[{{1, torch.floor(N / 2)}}],
    trainLabels = alllabels[{{1, torch.floor(N / 2)}}],
    testData = alldata[{{torch.floor(N / 2) + 1, N}}],
    testLabels = alllabels[{{torch.floor(N / 2) + 1, N}}]
}

local params = {}
params.model = {
    numItems = synthseg.MAX_LEN,
    itemDim = #synthseg.OBJECT + synthseg.MAX_ITEM_LEN,
    queryDim = #synthseg.OBJECT
}
params.loopConfig = {
    numEpoch = 10000,
    batchSize = 64,
    progressBar = true,
}
params.optimConfig = {
    learningRate = 0.1,
    learningRateDecay = 0.1 / (opt.num_ex / 2 / params.loopConfig.batchSize),
    weightDecay = 0.000005
}
local optimizer = optim.adam2
local model = {}
model.train = createModel(params.model)
model.eval = createModel(params.model)

local trainer = NNTrainer(
    model.train, params.loopConfig, optimizer, params.optimConfig)

local trainEval = NNEvaluator('train', model.eval)
local testEval = NNEvaluator('test', model.eval)

local visualize = function()
    local numTest = 10
    local testSubset = data.testData[{{1, numTest}}]
    local labelSubset = data.testLabels[{{1, numTest}}]
    local testOutput = model.eval:forward(testSubset)
    local dataDim = testSubset:size(2)
    print(dataDim)
    for i = 1, numTest do
        local _, ooi = torch.max(testSubset[i][{{dataDim - 3, dataDim}}], 1)
        io.write(string.format('Q: %2d\n', ooi[1]))
        io.write('D: ')
        local items = testSubset[i][{{1, dataDim - #synthseg.OBJECT}}]:reshape(
            params.model.numItems, params.model.itemDim)
        for j = 1, synthseg.MAX_LEN do
            local _, cat = torch.max(items[j][{{1, #synthseg.OBJECT}}], 1)
            local _, part = torch.max(items[j][{{1 + #synthseg.OBJECT, 
                #synthseg.OBJECT + synthseg.MAX_ITEM_LEN}}], 1)
            io.write(string.format('%2d(%1d)', cat[1], part[1]))
        end
        io.write('\nO: ')
        for j = 1, synthseg.MAX_LEN do
            io.write(string.format('%5.2f', testOutput[i][j][1]))
        end
        io.write('\nL: ')
        for j = 1, synthseg.MAX_LEN do
            io.write(string.format('%5.2f', labelSubset[i][j]))
        end
        io.write('\n')
    end
end

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
            if epoch % 1 == 0 and opt.viz then
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
