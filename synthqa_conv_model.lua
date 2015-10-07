local synthqa = require('synthqa')
local logger = require('logger')()
local nn = require('nn')
local nngraph = require('nngraph')
local rnn = require('rnn')
local lstm = require('lstm')
local constant = require('constant')
local batch_reshape = require('batch_reshape')
local gradient_stopper = require('gradient_stopper')
local counting_criterion = require('counting_criterion')
local double_counting_criterion = require('double_counting_criterion')
local attention_criterion = require('attention_criterion')
local lazy_gmodule = require('lazy_gmodule')
local synthqa_conv_model = {}

-------------------------------------------------------------------------------
function synthqa_conv_model.create(params, training)
    -- params.objectEmbedDim
    -- params.colorEmbedDim
    -- params.questionLength
    -- params.wordEmbedDim
    -- params.encoderDim
    -- params.recallerDim
    -- params.aggregatorDim
    -- params.inputItemDim
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
    local items = nn.Narrow(2, 1, params.numItems * params.inputItemDim)(input)
    local itemsReshape = nn.Reshape(
        params.numItems, params.inputItemDim)(items)
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
        2, params.numItems * params.inputItemDim + 1, params.questionLength)(
        input)
    local itemOfInterest = nn.Select(
        2, params.numItems * params.inputItemDim + 3)(input)
    local wordEmbed = nn.LookupTable(
        #synthqa.idict, params.wordEmbedDim)(
        mynn.GradientStopper()(wordIds))
    local wordEmbedSeq = nn.SplitTable(2)(wordEmbed)

    -- Encoder LSTM
    local constantEncoderState = mynn.Constant(params.encoderDim * 2, 0)(input)
    local encoderCore = lstm.createUnit(
        params.wordEmbedDim, params.encoderDim)
    local encoder = nn.RNN(
        encoderCore, params.questionLength)(
        {wordEmbedSeq, constantEncoderState})
    local encoderStateSel = nn.SelectTable(
        params.questionLength)(encoder)
    encoder.data.module.name = 'encoder'
    local encoderOutput = nn.Narrow(
        2, params.encoderDim + 1, params.encoderDim)(encoderStateSel)

    -- Filter
    local itemFilterLT = nn.Linear(
        params.encoderDim, params.itemDim)(encoderOutput)
    local itemFilterTH = mynn.BatchReshape(
        params.itemDim, 1)(nn.Tanh()(itemFilterLT))

    -- 1-D Convolution
    local attention = nn.MM()({itemsJoinedReshape, itemFilterTH})
    local attentionReshape = nn.Sigmoid()(
        mynn.BatchReshape(params.numItems)(attention))
    local final = nn.Sum(2)(attentionReshape)

    all = nn.LazyGModule({input}, {final})
    all:addModule('catEmbed', catEmbed)
    all:addModule('colorEmbed', colorEmbed)
    all:addModule('wordEmbed', wordEmbed)
    all:addModule('encoder', encoder)
    all:addModule('itemFilterLT', itemFilterLT)
    all:addModule('attention', attention)
    all:addModule('attentionReshape', attentionReshape)
    all:addModule('final', final)
    all:setup()

    -- Expand LSTMs
    encoder.data.module:expand()

    -- Criterion and decision function
    all.criterion = nn.MSECriterion()
    all.decision = function(pred)
        local num = torch.round(pred)
        return num
    end
    all.params = params

    return all
end

synthqa_conv_model.getVisualize = function(
    model, 
    attentionModule,
    data,
    rawData)
    return function()
        logger:logInfo('attention visualization')
        local numItems = model.params.numItems
        local outputTable = model:forward(data)
        for n = 1, data:size(1) do
            local rawDataItem = rawData[n]
            local realNumItems = 0
            local output
            if outputTable:size():size() == 2 then
                output = outputTable[n][1]
            else
                output = outputTable[n]
            end
            local itemsort, sortidx = data[n]:narrow(
                1, 1, numItems * model.params.inputItemDim):reshape(
                numItems, model.params.inputItemDim):select(
                2, model.params.inputItemDim):sort()
            print(string.format('%d. Q: %s (%d) A: %s O: %d', 
                n, rawDataItem.question, data[n][-1], 
                rawDataItem.answer, output))
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
            for i = 1, realNumItems do
                local idx = sortidx[i]
                io.write(string.format(
                    '%6.2f', attentionModule.output[n][idx]))
            end
            io.write('\n')
        end
    end
end
-------------------------------------------------------------------------------
return synthqa_conv_model
