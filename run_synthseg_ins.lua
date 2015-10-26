local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local logger = require('logger')()
local synthseg = require('synthseg')

local batch_reshape = require('batch_reshape')
local lazy_gmodule = require('lazy_gmodule')
local temporal_zero_padding = require('temporal_zero_padding')
local constant = require('constant')
local rnn = require('rnn')

local utils = require('utils')
local optim_pkg = require('optim')
local adam2 = require('adam')
local nntrainer = require('nntrainer')
local nnevaluator = require('nnevaluator')

-------------------------------------------------------------------------------
function createModel3(params)
    -- params.itemDim
    -- params.bn
    -- params.numItems
    -- params.filterWindowSizes
    -- params.numFilters
    -- params.timespan

    function convCore(inDepth, convSizes, numFilters)
        local input = nn.Identity()()
        local hiddenPrev = nn.Identity()()
        local semanticLabel = nn.Narrow(3, params.itemDim + 1, 1)(input)
        local image = nn.JoinTable(3)({hiddenPrev, input})
        local convs = {}
        local outDim = 0
        for i = 1, #convSizes do
            local pad = mynn.TemporalZeroPadding(
                        torch.floor((convSizes[i]  - 1) / 2), 
                        torch.ceil((convSizes[i] - 1) / 2))(image)
            local conv = nn.TemporalConvolution(
                    inDepth, numFilters[i], convSizes[i])(pad)
            conv.data.module.weight:uniform(-0.0001, 0.0001)
            conv.data.module.bias:uniform(-0.0001, 0.0001)
            table.insert(convs, conv)
            outDim = outDim + numFilters[i]
        end
        local join
        if #convSizes == 1 then
            join = convs[1]
        else
            join = nn.JoinTable(3)(convs)
        end 
        local relu = nn.ReLU()(join)
        local conv1 = nn.TemporalConvolution(
                outDim, 1, 1)(relu)
        conv1.data.module.weight:uniform(-0.0001, 0.0001)
        conv1.data.module.bias:uniform(-0.0001, 0.0001)
        local hiddenNext = nn.Sigmoid()(nn.CSubTable()({semanticLabel, conv1}))
        return nn.gModule({input, hiddenPrev}, {hiddenNext})
    end

    local input = nn.Identity()()
    local out = {input}
    local rnnCore = convCore(
        params.itemDim + 2, params.convSizes, params.numFilters)
    local constState = mynn.Constant({params.numItems, 1}, 0)(input)
    local inputList = {}
    for t = 1, params.timespan do
        table.insert(inputList, nn.Identity()(input))
    end
    local inputListNode = nn.Identity()(inputList)
    local rnn = nn.RNN(
        rnnCore, params.timespan)({inputListNode, constState})
    local final = nn.SelectTable(params.timespan)(rnn)
    local all = nn.LazyGModule({input}, {final})
    all:addModule('rnn', rnn)
    all:setup()
    all:expand()
    all.criterion = nn.BCECriterion()
    all.decision = function(pred)
        return torch.round(pred)
    end
    return all
end

-- -- -------------------------------------------------------------------------------
-- function createModel22(params)
--     -- params.itemDim
--     -- params.bn
--     -- params.numItems
--     -- params.filterWindowSizes (list)
--     -- params.numFilters (list)
--     -- params.padding (list)
--     -- params.strides (list)

--     local input = nn.Identity()()
--     -- Input is 3-d map
--     -- (B, NumItems, ItemDim + 1)
--     -- First plane is the item map
--     -- Second plane is the item semantic ground truth
--     local out = {input}
--     local prevNumFilters = params.itemDim + 1

--     local bns = {}
--     local convs = {}
--     local relus = {}
--     local semanticLabel = nn.Narrow(3, params.itemDim + 1, 1)(input)
--     local final
--     for i = 1, #params.numFilters do
--         local lastLayer = out[#out]
--         local curNumFilters = 0
--         local layerConv = {}
--         for j = 1, #params.numFilters[i] do
--             local numFilters = params.numFilters[i][j]
--             local convSize = params.filterWindowSizes[i][j]
--             local pad = mynn.TemporalZeroPadding(
--                     torch.floor((convSize  - 1) / 2), 
--                     torch.ceil((convSize - 1) / 2))(lastLayer)
--             print(prevNumFilters, numFilters, convSize)
--             local conv = nn.TemporalConvolution(
--                 prevNumFilters, numFilters, convSize)(pad)
--             conv.data.module.weight:uniform(-0.0001, 0.0001)
--             conv.data.module.bias:uniform(-0.0001, 0.0001)
--             table.insert(convs, conv)
--             table.insert(layerConv, conv)
--             curNumFilters = curNumFilters + numFilters
--         end
--         local join
--         if #layerConv < 2  then
--             join = layerConv
--         else
--             join = nn.JoinTable(3)(layerConv)
--         end
--         local relu = nn.ReLU()(join)
--         table.insert(relus, relu)
--         local convFinal = nn.TemporalConvolution(curNumFilters, 1, 1)(relu)
--         convFinal.data.module.weight:uniform(-0.0001, 0.0001)
--         convFinal.data.module.bias:uniform(-0.0001, 0.0001)
--         table.insert(convs, convFinal)
--         -- local sum = convFinal
--         local sum = nn.Sigmoid()(nn.CAddTable()({convFinal, semanticLabel}))
--         if i == #params.numFilters then
--             final = sum
--         end
--         out[#out + 1] = nn.JoinTable(3)({convFinal, input})
--         prevNumFilters = 1 + params.itemDim + 1
--     end

--     local all = nn.LazyGModule({input}, {final})
--     for i = 1, #convs do
--         all:addModule(string.format('conv_%d', i), convs[i])
--     end
--     all:setup()
    
--     all.criterion = nn.BCECriterion()
--     all.decision = function(pred)
--         return torch.round(pred)
--     end
--     return all
-- end

-- ----------------------------------------------------------------------------------
-- function createModel2(params)
--     -- params.itemDim
--     -- params.bn
--     -- params.numItems
--     -- params.filterWindowSizes (list)
--     -- params.numFilters (list)
--     -- params.padding (list)
--     -- params.strides (list)

--     local input = nn.Identity()()
--     -- Input is 3-d map
--     -- (B, NumItems, ItemDim + 1)
--     -- First plane is the item map
--     -- Second plane is the item semantic ground truth
--     local out = {input}
--     local prevNumFilters = params.itemDim + 1

--     local bns = {}
--     local convs = {}
--     local relus = {}
--     local semanticLabel = nn.Narrow(3, params.itemDim + 1, 1)(input)
--     local final
--     for i = 1, #params.numFilters do
--         local lastLayer = out[#out]
--         if params.bn then
--             local bn = nn.BatchNormalization(
--                 params.numItems * prevNumFilters)(
--                 mynn.BatchReshape(
--                     params.numItems * prevNumFilters)(out[#out]))
--             local bnReshape = mynn.BatchReshape(
--                 params.numItems, prevNumFilters)(bn)
--             table.insert(bns, bn)
--             lastLayer = bnReshape
--         end
--         local curNumFilters = 0
--         local layerConv = {}
--         for j = 1, #params.numFilters[i] do
--             local numFilters = params.numFilters[i][j]
--             local convSize = params.filterWindowSizes[i][j]
--             local pad = mynn.TemporalZeroPadding(
--                     torch.floor((convSize  - 1) / 2), 
--                     torch.ceil((convSize - 1) / 2))(lastLayer)
--             print(prevNumFilters, numFilters, convSize)
--             local conv = nn.TemporalConvolution(
--                 prevNumFilters, numFilters, convSize)(pad)
--             conv.data.module.weight:uniform(-0.0001, 0.0001)
--             conv.data.module.bias:uniform(-0.0001, 0.0001)
--             table.insert(convs, conv)
--             table.insert(layerConv, conv)
--             curNumFilters = curNumFilters + numFilters
--         end
--         local join
--         if #layerConv < 2  then
--             join = layerConv
--         else
--             join = nn.JoinTable(3)(layerConv)
--         end
--         local relu = nn.ReLU()(join)
--         table.insert(relus, relu)
--         local convFinal = nn.TemporalConvolution(curNumFilters, 1, 1)(relu)
--         convFinal.data.module.weight:uniform(-0.0001, 0.0001)
--         convFinal.data.module.bias:uniform(-0.0001, 0.0001)
--         table.insert(convs, convFinal)
--         -- local sum = convFinal
--         local sum = nn.CAddTable()({convFinal, semanticLabel})
--         if i == #params.numFilters then
--             final = sum
--         end
--         out[#out + 1] = nn.JoinTable(3)({convFinal, input})
--         prevNumFilters = 1 + params.itemDim + 1
--     end

--     local all = nn.LazyGModule({input}, {final})
--     if params.bn then
--         for i = 1, #bns do
--                 all:addModule(string.format('bn_%d', i), bns[i])
--         end
--     end
--     for i = 1, #convs do
--         all:addModule(string.format('conv_%d', i), convs[i])
--     end
--     all:setup()
    
--     all.criterion = nn.MarginCriterion()
--     all.decision = function(pred)
--         return pred:gt(0):float() * 2 - 1
--     end
--     return all
-- end

-------------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Synthetic Instance Segmentation Training')
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
local alldata = synthseg.encodeDataset2(dataset)
local alllabels = synthseg.getOneInstanceLabels(dataset)
-- alllabels = alllabels * 2 - 1

local data = {
    trainData = alldata[{{1, torch.floor(N / 2)}}],
    trainLabels = alllabels[{{1, torch.floor(N / 2)}}],
    testData = alldata[{{torch.floor(N / 2) + 1, N}}],
    testLabels = alllabels[{{torch.floor(N / 2) + 1, N}}]
}

local params = {}
-- params.model = {
--     numItems = synthseg.MAX_LEN,
--     itemDim = #synthseg.OBJECT + synthseg.MAX_ITEM_LEN,
--     -- numFilters = {20, 10, 1},
--     -- filterWindowSizes = {20, 20, 20}
--     -- numFilters = {5, 1},
--     -- filterWindowSizes = {20, 20}
--     numFilters = {3, 5, 10, 5, 3},
--     filterWindowSizes = {1, 10, 30, 10, 5},
--     padding = {{nil, nil},
--                 {nil, nil},
--                 {nil, nil},
--                 {nil, nil},
--                 {nil, nil}}
-- }
-- params.model = {
--     bn = true,
--     numItems = synthseg.MAX_LEN,
--     itemDim = #synthseg.OBJECT + synthseg.MAX_ITEM_LEN,
--     numFilters = {{10}, {10}, {10}, {10}, {10}},
--     filterWindowSizes = {{10}, {10}, {10}, {10}, {10}},
-- }
params.model = {
    numItems = synthseg.MAX_LEN,
    itemDim = #synthseg.OBJECT + synthseg.MAX_ITEM_LEN,
    numFilters = {100},
    convSizes = {30},
    timespan = 10
}
params.loopConfig = {
    numEpoch = 10000,
    batchSize = 20,
    progressBar = true,
}
params.optimConfig = {
    learningRate = 0.1,
    learningRateDecay = 0.1 / (opt.num_ex / 2 / params.loopConfig.batchSize),
    weightDecay = 0.00005
}
local optimizer = optim.adam2
local model = {}
model.train = createModel3(params.model)
model.eval = createModel3(params.model)
-- logger:logFatal(model.train.w:size())
params.optimConfig.gradientClip = utils.gradientClip(0.1, model.train.sliceLayer)


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
    for i = 1, 10 do
        local name = string.format('conv_%d', i)
        if model.train.moduleMap[name] then
            print(name, 'mean:', 
                model.train.moduleMap[name].output:mean())
            print(name, 'weight mean:', 
                model.train.moduleMap[name].weight:mean())
        end
    end
    -- print('conv final mean:', model.train.moduleMap['convFinal'].output:mean())
    -- print('conv final weight mean:', model.train.moduleMap['convFinal'].weight:mean())
    for i = 1, numTest do
        -- local _, ooi = torch.max(testSubset[i][{{dataDim - 3, dataDim}}], 1)
        -- io.write(string.format('Q: %2d\n', ooi[1]))
        io.write('D: ')
        local items = testSubset[{i, {}, {1, params.model.itemDim}}]
        -- print(items:size())
        for j = 1, synthseg.MAX_LEN do
            local _, cat = torch.max(items[j][{{1, #synthseg.OBJECT}}], 1)
            local _, part = torch.max(items[j][{{1 + #synthseg.OBJECT, 
                #synthseg.OBJECT + synthseg.MAX_ITEM_LEN}}], 1)
            io.write(string.format('%5d(%1d)', cat[1], part[1]))
        end
        io.write('\nS: ')
        for j = 1, synthseg.MAX_LEN do
            io.write(string.format('%8.2f', testSubset[{i, j, -1}]))
        end
        io.write('\nO: ')
        for j = 1, synthseg.MAX_LEN do
            io.write(string.format('%8.2f', testOutput[i][j][1]))
        end
        io.write('\nL: ')
        for j = 1, synthseg.MAX_LEN do
            io.write(string.format('%8.2f', labelSubset[i][j]))
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
