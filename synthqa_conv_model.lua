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
local synthqa_model =  require('synthqa_model')
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
    local inputProc = synthqa_model.createInputProc(params)(input)
    local itemsJoinedReshape = nn.SelectTable(1)(inputProc)
    local wordEmbedSeq = nn.SelectTable(2)(inputProc)

    -- Encoder LSTM
    local encoder = synthqa_model.createEncoderGRNN(
        params, wordEmbedSeq, input)
    local encoderStateSel = nn.SelectTable(params.questionLength)(encoder)

    -- Filter
    local attentionReshape, attentionModules = 
        synthqa_model.createConv1DAttentionUnit(
        params, encoderStateSel, itemsJoinedReshape)

    -- Sum
    local final = nn.Sum(2)(attentionReshape)

    all = nn.LazyGModule({input}, {final})
    all:addModule('inputProc', inputProc)
    all:addModule('encoder', encoder)
    all:addModuleMap(attentionModules)
    all:addModule('final', final)
    all:setup()
    all:expand()

    -- Criterion and decision function
    all.criterion = nn.MSECriterion()
    all.decision = function(pred)
        local num = torch.round(pred)
        return num
    end
    all.params = params

    return all
end

-------------------------------------------------------------------------------
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
