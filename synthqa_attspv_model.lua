local synthqa = require('synthqa')
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
local synthqa_attspv_model = {}

-------------------------------------------------------------------------------
function synthqa_attspv_model.create(params, training)
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
    local gridId = nn.SelectTable(3)(inputProc)
    local itemOfInterest = nn.SelectTable(4)(inputProc)
    local catIdReshape = nn.SelectTable(5)(inputProc)

    -- Question encoder
    local encoder = synthqa_model.createEncoderLSTM(params, wordEmbedSeq, input)
    local encoderStateSel = nn.SelectTable(params.questionLength)(encoder)

    -- Attention decoder
    -- local decoderCore = synthqa_model.createLSTMAttentionController(
    --     1, params.encoderDim, params.numItems, params.itemDim, 0.1, 
    --     params.attentionMechanism, training)
    local decoderCore = synthqa_model.createLSTMAttentionController(
        params, training)

    local decoder = synthqa_model.createDecoderLSTM(
        params, decoderCore, encoderStateSel, itemsJoinedReshape)

    -- Attended item at every timestep (split, T, N x M)
    -- Attention value with penalty (joined, N x T x M)
    -- Attention value hard (joined, N x T x M)
    -- Attention value soft (joined, N x T x M)
    -- Attention value selected (joined, N x T)
    attentionOutputTable, penAttentionValueReshape, hardAttentionValueReshape, 
        softAttentionValueReshape, softAttentionSelJoin = 
        synthqa_model.decoderOutputs(params, decoder)

    -- Recaller
    local recaller, recallerOutputMap, recallerBinaryReshape, recallerAttMul, 
        recallerBinarySplit = synthqa_model.createRecallerLSTM(
        params, attentionOutputTable, softAttentionSelJoin, input)

    -- Aggregator (adds 1's and 0's)
    local aggregator, outputMap, outputMapReshape = 
        synthqa_model.createAggregatorLSTM(
        params, recallerBinarySplit, input)

    -- Model outputs
    local final = nn.Select(2, params.decoderSteps)(outputMapReshape)
    local countingCriterionOutput = nn.Identity()(
        {
            outputMapReshape,
            recallerAttMul
        })
    local attentionCriterionOutput = nn.Identity()(
        {
            softAttentionValueReshape, 
            hardAttentionValueReshape,
            itemOfInterest, 
            catIdReshape,
            penAttentionValueReshape
        })
    local doubleCountingCriterionOutput = nn.Identity()(
        {
            recallerBinaryReshape, 
            gridId, 
            hardAttentionValueReshape
        })

    -- Build model
    local all = nn.LazyGModule({input}, 
        {
            final, 
            countingCriterionOutput,
            attentionCriterionOutput, 
            doubleCountingCriterionOutput
        })

    all:addModule('inputProc', inputProc)
    all:addModule('encoder', encoder)
    all:addModule('decoder', decoder)
    all:addModule('recaller', recaller)
    all:addModule('recallerOutputMap', recallerOutputMap)
    all:addModule('recallerBinaryReshape', recallerBinaryReshape)
    all:addModule('softAttentionSelJoin', softAttentionSelJoin)
    all:addModule('softAttentionValueReshape', softAttentionValueReshape)
    all:addModule('hardAttentionValueReshape', hardAttentionValueReshape)
    all:addModule('penAttentionValueReshape', penAttentionValueReshape)
    all:addModule('aggregator', aggregator)
    all:addModule('outputMap', outputMap)
    all:addModule('final', final)
    all:setup()
    all:expand()

    -- Criterion and decision function
    all.criterion = nn.ParallelCriterion(true)
      :add(nn.MSECriterion(), 0.0)
      :add(mynn.CountingCriterion(), 0.1)
      :add(mynn.AttentionCriterion(), 1.0)
      :add(mynn.DoubleCountingCriterion(), 1.0)
    all.decision = function(pred)
        local num = torch.round(pred)
        return num
    end

    -- Store parameters
    all.params = params

    return all
end

-------------------------------------------------------------------------------
synthqa_attspv_model.gradientClip = {
    inputProc = 0.1,
    encoder = 0.1,
    decoder = 0.1,
    recaller = 0.1,
    recallerOutputMap = 0.1,
    aggregator = 0.1,
    outputMap = 0.1
}

-------------------------------------------------------------------------------
return synthqa_attspv_model
