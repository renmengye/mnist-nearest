local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local synthqa = require('synthqa')
local lstm = require('lstm')
local gru = require('gru')
local synthqa_model = {}

-------------------------------------------------------------------------------
function synthqa_model.createInputProc(params)
    -- params.objectEmbedDim
    -- params.colorEmbedDim
    -- params.questionLength
    -- params.wordEmbedDim
    -- params.inputItemDim
    -- params.itemDim
    -- params.numItems
    -- params.vocabSize
    -- params.numObject
    -- params.numColor

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

    -- Ground truth for training the memory recall layer.
    local gridId = nn.Select(3, 7)(itemsReshape)

    -- Item embeddings
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

    -- Word embeddings
    local wordIds = nn.Narrow(
        2, params.numItems * params.inputItemDim + 1, params.questionLength)(
        input)
    local itemOfInterest = nn.Select(
        2, params.numItems * params.inputItemDim + 3)(input)
    local wordEmbed = nn.LookupTable(
        #synthqa.idict, params.wordEmbedDim)(
        mynn.GradientStopper()(wordIds))
    local wordEmbedSeq = nn.SplitTable(2)(wordEmbed)
    local inputLayer = nn.gModule({input}, 
        {
            itemsJoinedReshape, 
            wordEmbedSeq, 
            gridId, 
            itemOfInterest, 
            catIdReshape
        })

    inputLayer.moduleMap = {
        catEmbed = catEmbed.data.module,
        colorEmbed = colorEmbed.data.module,
        wordEmbed = wordEmbed.data.module
    }
    return inputLayer
end

-------------------------------------------------------------------------------
function synthqa_model.createEncoderLSTM(params, wordEmbedSeq, input)
    local constantEncoderState = mynn.Constant(params.encoderDim * 2, 0)(input)
    local encoderCore = lstm.createUnit(
        params.wordEmbedDim, params.encoderDim)
    local encoder = nn.RNN(
        encoderCore, params.questionLength)(
        {wordEmbedSeq, constantEncoderState})
    encoder.data.module.name = 'encoder'
    return encoder
end

-------------------------------------------------------------------------------
function synthqa_model.createEncoderGRNN(params, wordEmbedSeq, input)
    local constantEncoderState = mynn.Constant(params.encoderDim, 0)(input)
    local encoderCore = gru.createUnit(
        params.wordEmbedDim, params.encoderDim)
    local encoder = nn.RNN(encoderCore, params.questionLength)(
        {wordEmbedSeq, constantEncoderState})
    encoder.data.module.name = 'encoder'
    return encoder
end

-------------------------------------------------------------------------------
function synthqa_model.createDecoderRNN(
    params, decoderCore, encoderStateSel, itemsJoinedReshape)
    local constantDecoderState = mynn.Constant(
        params.itemDim + params.numItems * 4 + 1)(encoderStateSel)
    local decoderStateInit = nn.JoinTable(2)(
        {encoderStateSel, constantDecoderState})
    decoderStateInit.data.module.name = 'decoderStateInit'
    local decoderInputConst = mynn.Constant(
        {params.decoderSteps, 1}, 0)(encoderStateSel)
    local decoderInputSplit = nn.SplitTable(2)(decoderInputConst)
    local decoder = nn.RNN(
        decoderCore, params.decoderSteps)(
        {decoderInputSplit, decoderStateInit, itemsJoinedReshape})
    decoder.data.module.name = 'decoder'

    return decoder
end

-------------------------------------------------------------------------------
function synthqa_model.createDecoderGRNNOutputs(params, decoder)
    local attentionOutputs = {}
    local softAttentionValues = {}
    local hardAttentionValues = {}
    local penAttentionValues = {}
    local softAttentionSel = {}
    for t = 1, params.decoderSteps do
        table.insert(attentionOutputs, nn.Narrow(
            2, 
            params.encoderDim + 1, 
            params.itemDim)(
            nn.SelectTable(t)(decoder)))
        table.insert(softAttentionValues, nn.Narrow(
            2, 
            params.encoderDim + 
            params.itemDim + 1, 
            params.numItems)(
            nn.SelectTable(t)(decoder)))
        table.insert(hardAttentionValues, nn.Narrow(
            2, params.encoderDim + 
            params.itemDim + 
            params.numItems + 1,
            params.numItems)(
            nn.SelectTable(t)(decoder)))
        table.insert(penAttentionValues, nn.Narrow(
            2, params.encoderDim + 
            params.itemDim + 
            params.numItems * 3 + 1,
            params.numItems)(
            nn.SelectTable(t)(decoder)))
        table.insert(softAttentionSel, nn.Narrow(
            2, 
            params.encoderDim + 
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

    return attentionOutputTable, penAttentionValueReshape, 
    hardAttentionValueReshape, softAttentionValueReshape, softAttentionSelJoin
end

-------------------------------------------------------------------------------
function synthqa_model.createDecoderLSTMOutputs(params, decoder)
    -- Attended item at every timestep (split, T, N x M)
    -- Attention value with penalty (joined, N x T x M)
    -- Attention value hard (joined, N x T x M)
    -- Attention value soft (joined, N x T x M)
    -- Attention value selected (joined, N x T)
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

    return attentionOutputTable, penAttentionValueReshape, 
    hardAttentionValueReshape, softAttentionValueReshape, softAttentionSelJoin
end

-------------------------------------------------------------------------------
function synthqa_model.createRecallerLSTM(
    params, attentionOutputTable, softAttentionSelJoin, input)
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

    return recaller, recallerOutputMap, recallerBinaryReshape, 
        recallerAttMul, recallerBinarySplit
end

-------------------------------------------------------------------------------
function synthqa_model.createAggregatorLSTM(params, recallerBinarySplit, input)
    local constantAggState = mynn.Constant(params.aggregatorDim * 2, 0)(input)
    local aggregator = nn.RNN(
        lstm.createUnit(1, params.aggregatorDim), params.decoderSteps)(
        {recallerBinarySplit, constantAggState})
    aggregator.data.module.name = 'aggregator'
    -- if params.aggregatorWeights then
    --     local agg_w, agg_dl_dw = aggregator.data.module.core:getParameters()
    --     agg_w:copy(params.aggregatorWeights)
    -- end

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

    -- -- Load pretrained weights
    -- if params.outputMapWeights then
    --     local ow, odldw = outputMap.data.module:getParameters()
    --     ow:copy(params.outputMapWeights)
    -- end
    return aggregator, outputMap, outputMapReshape

end

-------------------------------------------------------------------------------
function synthqa_model.createLSTMAttentionController(params, initRange, training)
    local inputDim = 1
    local hiddenDim = params.encoderDim
    local numItems = params.numItems
    local itemDim = params.itemDim
    if initRange == nil then
        initRange = 0.1
    end
    if training == nil then
        training = true
    end
    local input = nn.Identity()()
    local statePrev = nn.Identity()()
    -- batch x numItems x itemDim
    local items = nn.Identity()()
    local cellPrev = nn.Narrow(2, 1, hiddenDim)(statePrev)
    local hiddenPrev = nn.Narrow(2, hiddenDim + 1, hiddenDim)(statePrev)
    local attentionAccumPrev = nn.Narrow(2, 
        hiddenDim * 2 + 
        itemDim + 
        numItems * 2 + 1, 
        numItems)(statePrev)

    -- Attention
    local attentionSig, attentionModule = 
        synthqa_model.createConv1DAttentionUnit(params, hiddenPrev, items)

    -- Select item based on attention
    local attentionSelModules, attentionSelMap = 
        synthqa_model.createAttentionSelector(
        params, attentionAccumPrev, attentionSig, items, input)
    local attendedItems, attentionSigSel, hardAttention = 
        unpack(attentionSelModules)
    local attentionAccum = attentionSelMap['attentionAccum']
    local attentionSigPenNorm = attentionSelMap['attentionSigPenNorm']

    -- LSTM controller
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
         attentionSig,
         hardAttention,
         attentionAccum,
         attentionSigPenNorm,
         attentionSigSel})
    stateNext.data.module.name = 'attStateNext'
    local coreModule = nn.gModule({input, statePrev, items}, {stateNext})

    -- coreModule.reinforceUnit = attention.data.module
    coreModule.moduleMap = {
        softAttention = attentionSigPenNorm.data.module,
        hardAttention = hardAttention.data.module
    }
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
function synthqa_model.createGRNNAttentionController(
    params, initRange, inGateBiasInit, training)
    local inputDim = 1
    local hiddenDim = params.encoderDim
    local numItems = params.numItems
    local itemDim = params.itemDim
    local initRange = 0.1
    local training = true

    if initRange == nil then
        initRange = 0.1
    end
    if training == nil then
        training = false
    end
    local input = nn.Identity()()
    local statePrev = nn.Identity()()
    -- batch x numItems x itemDim
    local items = nn.Identity()()
    local hiddenPrev = nn.Narrow(2, 1, hiddenDim)(statePrev)
    local attentionAccumPrev = nn.Narrow(2, 
        hiddenDim + 
        itemDim + 
        numItems * 2 + 1, 
        numItems)(statePrev)

    -- Attention
    local attentionSig, attentionModule = 
        synthqa_model.createConv1DAttentionUnit(params, hiddenPrev, items)

    -- Select item based on attention
    local attentionSelModules, attentionSelMap = 
        synthqa_model.createAttentionSelector(
        params, attentionAccumPrev, attentionSig, items, input)
    local attendedItems, attentionSigSel, hardAttention = 
        unpack(attentionSelModules)
    local attentionAccum = attentionSelMap['attentionAccum']
    local attentionSigPenNorm = attentionSelMap['attentionSigPenNorm']

    -- GRNN controller    
    local joinState = nn.JoinTable(2)({input, hiddenPrev})
    local joinStateDim = inputDim + hiddenDim
    local inGateLT = nn.Linear(joinStateDim, hiddenDim)(joinState)
    local inGate = nn.Sigmoid()(inGateLT)
    local transformLT = nn.Linear(joinStateDim, hiddenDim)(joinState)
    local transform = nn.Tanh()(transformLT)
    local ones = mynn.Constant(hiddenDim, 1)(input)
    local carryGate = nn.CSubTable()({ones, inGate})
    local hiddenNext = nn.CAddTable()({
        nn.CMulTable()({inGate, transform}),
        nn.CMulTable()({carryGate, hiddenPrev})
    })
    local stateNext = nn.JoinTable(2)(
        {hiddenNext,
         attendedItems,
         attentionSig,
         hardAttention,
         attentionAccum,
         attentionSigPenNorm,
         attentionSigSel})
    stateNext.data.module.name = 'attStateNext'

    local coreModule = nn.gModule({input, statePrev, items}, {stateNext})
    coreModule.moduleMap = {
        inGateLT = inGateLT.data.module,
        inGate = inGate.data.module,
        transformLT = transformLT.data.module,
        transform = transform.data.module,
        carryGate = carryGate.data.module
    }
    joinState.data.module.name = 'gru.joinState'
    inGateLT.data.module.name = 'gru.inGateLT'
    transformLT.data.module.name = 'gru.transformLT'
    inGateLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLT.data.module.bias:fill(inGateBiasInit)
    transformLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    transformLT.data.module.bias:fill(0)

    return coreModule
end

-------------------------------------------------------------------------------
function synthqa_model.createAttentionSelector(
    params, attentionAccumPrev, attentionSig, items, input)
    local transitionOut = mynn.Constant({params.numItems}, 0.5)(input)
    local ones = mynn.Constant({params.numItems}, 1.0)(input)
    local stayPenalty = nn.CMulTable()({transitionOut, attentionAccumPrev})
    local stayPenaltyCoeff = nn.CSubTable()({ones, stayPenalty})
    local attentionSigPen = nn.CMulTable()({attentionSig, stayPenaltyCoeff})
    local attentionSigPenSum = nn.Replicate(
        params.numItems, 2)(nn.Sum(2)(attentionSigPen))
    local attentionSigPenNorm = nn.CDivTable()(
        {attentionSigPen, attentionSigPenSum})

    local attentionIdx = nn.ArgMax(2)(attentionSigPenNorm)
    local hardAttention = mynn.OneHot(params.numItems)(attentionIdx)
    local attentionAccum = nn.CAddTable()({hardAttention, stayPenalty})
    local attentionReshape = nn.Reshape(1, params.numItems)(hardAttention)
    local mm2 = nn.MM()({attentionReshape, items})
    mm2.data.module.name = 'MM2'
    local attendedItems = nn.Reshape(params.itemDim)(mm2)
    local attentionSigSel = mynn.BatchReshape(1)(
        nn.Sum(2)(nn.CMulTable()({attentionSig, hardAttention})))

    local moduleMap = {
        attentionAccum = attentionAccum,
        attentionSigPenNorm = attentionSigPenNorm
    }
    return {attendedItems, attentionSigSel, hardAttention}, moduleMap
end

-------------------------------------------------------------------------------
function synthqa_model.createMLPAttentionUnit(params, hidden, items)
    local hiddenPrevLT = nn.Linear(
        params.encoderDim, params.itemDim)(hidden)
    local hiddenPrevLTRepeat = nn.Replicate(params.numItems, 2)(hiddenPrevLT)
    local itemsReshape = mynn.BatchReshape(params.itemDim)(items)
    local itemsLT = nn.Linear(params.itemDim, params.itemDim)(itemsReshape)
    local itemsLTReshape = mynn.BatchReshape(
        params.numItems, params.itemDim)(itemsLT)

    -- Query
    local query = nn.Tanh()(
        nn.CAddTable()({hiddenPrevLTRepeat, itemsLTReshape}))
    local attWeight = mynn.Weights(params.itemDim)(hidden)
    local attentionWeight = nn.Reshape(params.itemDim, 1)(attWeight)

    -- 1-D Convolution
    local attention = nn.MM()({query, attentionWeight})
    local attentionReshape = nn.Sigmoid()(
        mynn.BatchReshape(params.numItems)(attention))

    if params.hiddenPrevLTWeights then
        local w, dl_dw = hiddenPrevLT.data.module:getParameters()
        w:copy(params.hiddenPrevLTWeights)
    end
    if params.itemsLTWeights then
        local w, dl_dw = itemsLT.data.module:getParameters()
        w:copy(params.itemsLTWeights)
    end

    local moduleMap = {
        hiddenPrevLT = hiddenPrevLT,
        itemsLT = itemsLT,
        attWeight = attWeight,
        attention = attention,
        attentionReshape = attentionReshape
    }
    return attentionReshape, moduleMap
end

-------------------------------------------------------------------------------
function synthqa_model.createConv1DAttentionUnit(params, hidden, items)
    -- Filter
    local itemFilterLT = nn.Linear(
        params.encoderDim, params.itemDim)(hidden)
    local itemFilterTH = mynn.BatchReshape(
        params.itemDim, 1)(nn.Tanh()(itemFilterLT))

    -- 1-D Convolution
    local attention = nn.MM()({items, itemFilterTH})
    local attentionReshape = nn.Sigmoid()(
        mynn.BatchReshape(params.numItems)(attention))
    if params.itemFilterLTWeights then
        local w, dl_dw = itemFilterLT.data.module:getParameters()
        w:copy(params.itemFilterLTWeights)
    end
    local moduleMap = {
        itemFilterLT = itemFilterLT,
        attention = attention,
        attentionReshape = attentionReshape
    }
    return attentionReshape, moduleMap
end

return synthqa_model
