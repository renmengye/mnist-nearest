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
local synthqa_attspv_model = {}

-------------------------------------------------------------------------------
function synthqa_attspv_model.createAttentionUnit(
    inputDim,
    hiddenDim,
    numItems,
    itemDim,
    initRange,
    attentionMechanism,
    training)
    if initRange == nil then
        initRange = 0.1
    end
    -- Soft or hard attention
    if attentionMechanism == nil then
        attentionMechanism = 'soft'
    end
    if training == nil then
        training = false
    end
    local input = nn.Identity()()

    local statePrev = nn.Identity()()
    -- batch x numItems x itemDim
    local items = nn.Identity()()
    local cellPrev = nn.Narrow(2, 1, hiddenDim)(statePrev)
    local hiddenPrev = nn.Narrow(2, hiddenDim + 1, hiddenDim)(statePrev)
    local attentionAccumPrev = nn.Narrow(
        2, 
        hiddenDim * 2 + 
        itemDim + 
        numItems * 2 + 
        1, 
        numItems)(statePrev)

    local itemsReshape = mynn.BatchReshape(itemDim)(items)
    itemsReshape.data.module.name = 'itemsReshape'
    local hiddenPrevLT = nn.Linear(hiddenDim, itemDim)(hiddenPrev)
    local hiddenPrevLTRepeat = nn.Replicate(numItems, 2)(hiddenPrevLT)

    local itemsLT = nn.Linear(itemDim, itemDim)(itemsReshape)
    local itemsLTReshape = mynn.BatchReshape(numItems, itemDim)(itemsLT)
    itemsLTReshape.data.module.name = 'itemsLTReshape'

    local query = nn.Tanh()(
        nn.CAddTable()({hiddenPrevLTRepeat, itemsLTReshape}))

    local attWeight = mynn.Weights(itemDim)(input)
    local attentionWeight = nn.Reshape(itemDim, 1)(attWeight)
    attentionWeight.data.module.name = 'attentionWeight'
    local mm1 = nn.MM()({query, attentionWeight})
    mm1.data.module.name = 'MM1'
    local attentionUnnorm = nn.Reshape(numItems)(mm1)
    attentionUnnorm.data.module.name = 'attentionUnnorm'
    -- local attentionNorm = nn.SoftMax()(attentionUnnorm)
    local attentionNorm = nn.Sigmoid()(attentionUnnorm)

    local transitionOut = mynn.Constant({numItems}, 0.5)(input)
    local ones = mynn.Constant({numItems}, 1.0)(input)
    local stayPenalty = nn.CMulTable()({transitionOut, attentionAccumPrev})
    local stayPenaltyCoeff = nn.CSubTable()({ones, stayPenalty})
    local attentionNormPen = nn.CMulTable()({attentionNorm, stayPenaltyCoeff})
    local attentionNormPenSum = nn.Replicate(
        numItems, 2)(nn.Sum(2)(attentionNormPen))
    local attentionNormPenNorm = nn.CDivTable()(
        {attentionNormPen, attentionNormPenSum})

    -- Build attention using either 'hard' or 'soft' attention
    local attention
    if attentionMechanism == 'hard' then
        -- attention = nn.ReinforceCategorical(training)(attentionNormPenNorm)
        attentionIdx = nn.ArgMax(2)(attentionNormPenNorm)
        attention = mynn.OneHot(numItems)(attentionIdx)
    elseif attentionMechanism == 'soft' then
        attention = attentionNormPen
        -- logger:logFatal('no soft')
    else
        logger:logFatal(string.format(
            'unknown attention mechanism: %s', attentionMechanism))
    end

    local attentionAccum = nn.CAddTable()({attention, stayPenalty})
    local attentionReshape = nn.Reshape(1, numItems)(attention)
    local mm2 = nn.MM()({attentionReshape, items})
    mm2.data.module.name = 'MM2'
    local attendedItems = nn.Reshape(itemDim)(mm2)

    local attentionNormSel = mynn.BatchReshape(1)(
        nn.Sum(2)(nn.CMulTable()({attentionNorm, attention})))

    local joinState1 = nn.JoinTable(2)(
        {input, attendedItems, cellPrev, hiddenPrev})
    joinState1.data.module.name = 'attJoinState1'
    local joinState2 = nn.JoinTable(2)({input, attendedItems, hiddenPrev})
    joinState2.data.module.name = 'attJoinState2'
    local state1 = hiddenDim * 2 + inputDim + itemDim
    local state2 = hiddenDim + inputDim + itemDim
    local inGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local inGate = nn.Sigmoid()(inGateLin)
    local inTransformLin = nn.Linear(state2, hiddenDim)(joinState2)
    local inTransform = nn.Tanh()(inTransformLin)
    local forgetGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local forgetGate = nn.Sigmoid()(forgetGateLin)
    local outGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local outGate = nn.Sigmoid()(outGateLin)
    local cellNext = nn.CAddTable()({
        nn.CMulTable()({forgetGate, cellPrev}),
        nn.CMulTable()({inGate, inTransform})
    })
    local hiddenNext = nn.CMulTable()({outGate, nn.Tanh()(cellNext)})
    local stateNext = nn.JoinTable(2)(
        {cellNext, 
         hiddenNext,
         attendedItems,
         attentionNorm,
         attention,
         attentionAccum,
         attentionNormPenNorm,
         attentionNormSel})
    stateNext.data.module.name = 'attStateNext'
    local coreModule = nn.gModule({input, statePrev, items}, {stateNext})
    -- coreModule.reinforceUnit = attention.data.module
    coreModule.moduleMap = {
        softAttention = attentionNormPenNorm.data.module,
        attention = attention.data.module
    }
    attWeight.data.module.weight:uniform(-initRange / 2, initRange / 2)
    itemsLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    hiddenPrevLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.bias:fill(1)
    inTransformLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inTransformLin.data.module.bias:fill(0)
    forgetGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    forgetGateLin.data.module.bias:fill(1)
    outGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    outGateLin.data.module.bias:fill(1)

    return coreModule
end

-------------------------------------------------------------------------------
function synthqa_attspv_model.create(params, training)
    -- params.objectEmbedDim
    -- params.colorEmbedDim
    -- params.questionLength
    -- params.wordEmbedDim
    -- params.encoderDim
    -- params.recallerDim
    -- params.aggregatorDim
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
    local items = nn.Narrow(2, 1, params.numItems * 6)(input)
    local itemsReshape = nn.Reshape(params.numItems, 6)(items)
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
        2, params.numItems * 6 + 1, params.questionLength)(input)
    local itemOfInterest = nn.Select(2, params.numItems * 6 + 3)(input)
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

    -- Decoder LSTM (Attend to the object of interest)
    local constantDecoderState = mynn.Constant(
        params.itemDim + params.numItems * 4 + 1)(input)
    local decoderStateInit = nn.JoinTable(2)(
        {encoderStateSel, constantDecoderState})
    decoderStateInit.data.module.name = 'decoderStateInit'
    local decoderInputConst = mynn.Constant(
        {params.decoderSteps, 1}, 0)(input)
    local decoderInputSplit = nn.SplitTable(2)(decoderInputConst)

    -- Decoder share states with encoder
    local decoderCore = synthqa_attspv_model.createAttentionUnit(
        1, params.encoderDim, params.numItems, params.itemDim, 0.1, 
        params.attentionMechanism, training)
    local decoder = nn.RNN(
        decoderCore, params.decoderSteps)(
        {decoderInputSplit, decoderStateInit, itemsJoinedReshape})
    decoder.data.module.name = 'decoder'

    -- Gather decoder outputs
    local attentionOutputs = {}
    local softAttentionValues = {}
    local hardAttentionValues = {}
    local penAttentionValues = {}
    local softAttentionSel = {}
    for t = 1, params.decoderSteps do
        table.insert(attentionOutputs, nn.Narrow(
            2, 
            params.encoderDim * 2 + 1, 
            params.itemDim)(
            nn.SelectTable(t)(decoder)))
        table.insert(softAttentionValues, nn.Narrow(
            2, 
            params.encoderDim * 2 + 
            params.itemDim + 1, 
            params.numItems)(
            nn.SelectTable(t)(decoder)))
        table.insert(hardAttentionValues, nn.Narrow(
            2, params.encoderDim * 2 + 
            params.itemDim + 
            params.numItems + 1,
            params.numItems)(
            nn.SelectTable(t)(decoder)))
        table.insert(penAttentionValues, nn.Narrow(
            2, params.encoderDim * 2 + 
            params.itemDim + 
            params.numItems * 3 + 1,
            params.numItems)(
            nn.SelectTable(t)(decoder)))
        table.insert(softAttentionSel, nn.Narrow(
            2, 
            params.encoderDim * 2 + 
            params.itemDim + 
            params.numItems * 4 + 1, 
            1)(
            nn.SelectTable(t)(decoder)))
    end
    local softAttentionValueJoin = nn.JoinTable(2)(softAttentionValues)
    softAttentionValueJoin.data.module.name = 'softAttentionValueJoin'
    local softAttentionValueReshape = mynn.BatchReshape(
        params.decoderSteps, params.numItems)(softAttentionValueJoin)
    local hardAttentionValueJoin = nn.JoinTable(2)(hardAttentionValues)
    hardAttentionValueJoin.data.module.name = 'hardAttentionValueJoin'
    local hardAttentionValueReshape = mynn.BatchReshape(
        params.decoderSteps, params.numItems)(hardAttentionValueJoin)
    local penAttentionValueJoin = nn.JoinTable(2)(penAttentionValues)
    penAttentionValueJoin.data.module.name = 'penAttentionValueJoin'
    local penAttentionValueReshape = mynn.BatchReshape(
        params.decoderSteps, params.numItems)(penAttentionValueJoin)
    local attentionOutputTable = nn.Identity()(attentionOutputs)
    local softAttentionSelJoin = mynn.BatchReshape(params.decoderSteps, 1)(
        nn.JoinTable(2)(softAttentionSel))

    -- Recaller LSTM (Tell whether you have seen the object before)
    local recallerStateInit = mynn.Constant(params.recallerDim * 2, 0)(input)
    local recaller = nn.RNN(
        lstm.createUnit(
            params.itemDim, params.recallerDim), params.decoderSteps)(
        {attentionOutputTable, recallerStateInit})
    recaller.data.module.name = 'recaller'

    -- Recaller binary output
    local recallerOutputs = {}
    for t = 1, params.decoderSteps do
        table.insert(recallerOutputs, nn.Narrow(
            2, params.recallerDim + 1, params.recallerDim)(
            nn.SelectTable(t)(recaller)))
    end
    local recallerJoin = mynn.BatchReshape(params.recallerDim)(
        nn.JoinTable(2)(recallerOutputs))
    local recallerOutputMap = nn.Linear(params.recallerDim, 1)(recallerJoin)
    local recallerBinary = nn.Sigmoid()(recallerOutputMap)
    local recallerBinaryReshape = mynn.BatchReshape(params.decoderSteps, 1)(
        recallerBinary)

    -- Multiply with attention
    local recallerAttMul = nn.CMulTable()(
            {recallerBinaryReshape, softAttentionSelJoin})
    local recallerBinarySplit = nn.SplitTable(2)(recallerAttMul)

    -- Aggregator (adds 1's and 0's)
    local constantAggState = mynn.Constant(params.aggregatorDim * 2, 0)(input)
    local aggregator = nn.RNN(
        lstm.createUnit(1, params.aggregatorDim), params.decoderSteps)(
        {recallerBinarySplit, constantAggState})
    
    -- Load pretrained weights
    if params.aggregatorWeights then
        local agg_w, agg_dl_dw = aggregator.data.module.core:getParameters()
        agg_w:copy(params.aggregatorWeights)
    end
    aggregator.data.module.name = 'aggregator'

    -- Classify answer
    local aggregatorOutputs = {}
    for t = 1, params.decoderSteps do
        table.insert(aggregatorOutputs, nn.Narrow(
            2, params.aggregatorDim + 1, params.aggregatorDim)(
            nn.SelectTable(t)(aggregator)))
    end
    local aggregatorOutputJoin = nn.JoinTable(2)(aggregatorOutputs)
    aggregatorOutputJoin.data.module.name = 'aggregatorOutputJoin'
    local aggregatorOutputReshape = mynn.BatchReshape(
        params.aggregatorDim)(aggregatorOutputJoin)
    local outputMap = nn.Linear(params.aggregatorDim, 1)(
        aggregatorOutputReshape)
    local outputMapReshape = mynn.BatchReshape(
        params.decoderSteps, 1)(outputMap)

    -- Load pretrained weights
    if params.outputMapWeights then
        local ow, odldw = outputMap.data.module:getParameters()
        ow:copy(params.outputMapWeights)
    end
    local final = nn.Select(2, params.decoderSteps)(outputMapReshape)

    local attentionOut = nn.Identity()(
        {softAttentionValueReshape, 
         hardAttentionValueReshape,
         itemOfInterest, 
         catIdReshape,
         penAttentionValueReshape})

    all = nn.LazyGModule({input}, 
        {final, outputMapReshape, attentionOut, 
        recallerBinaryReshape})

    all:addModule('catEmbed', catEmbed)
    all:addModule('colorEmbed', colorEmbed)
    all:addModule('wordEmbed', wordEmbed)
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

    -- Expand LSTMs
    encoder.data.module:expand()
    decoder.data.module:expand()
    recaller.data.module:expand()
    aggregator.data.module:expand()

    -- Criterion and decision function
    all.criterion = nn.ParallelCriterion(true)
      :add(nn.MSECriterion(), 0.0)
      :add(mynn.CountingCriterion(recallerAttMul.data.module), 0.1)
      :add(mynn.AttentionCriterion(decoder.data.module), 1.0)
      :add(mynn.DoubleCountingCriterion(decoder.data.module), 1.0)
    all.decision = function(pred)
        local num = torch.round(pred)
        return num
    end

    return all
end

-------------------------------------------------------------------------------
-- synthqa_attspv_model.learningRates = {
--     catEmbed = 0.1,
--     colorEmbed = 0.1,
--     wordEmbed = 0.1,
--     encoder = 0.1,
--     decoder = 0.1,
--     recaller = 0.1,
--     recallerOutputMap = 0.1,
--     aggregator = 0.1,
--     outputMap = 0.1,
-- }

synthqa_attspv_model.gradClipTable = {
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
-------------------------------------------------------------------------------
return synthqa_attspv_model
