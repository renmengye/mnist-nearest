local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local weights = require('weights')
local batch_reshape = require('batch_reshape')
local logger = require('logger')()
local dpnn = require('dpnn')
local one_hot = require('one_hot')
local lstm = {}

-------------------------------------------------------------------------------
function lstm.createUnit(inputDim, hiddenDim, initRange)
    if initRange == nil then
        initRange = 0.1
    end
    local input = nn.Identity()()
    local statePrev = nn.Identity()()
    local cellPrev = nn.Narrow(2, 1, hiddenDim)(statePrev)
    local hiddenPrev = nn.Narrow(2, hiddenDim + 1, hiddenDim)(statePrev)
    local joinState1 = nn.JoinTable(2, 2)({input, statePrev})
    joinState1.data.module.name = 'joinState1'
    local joinState2 = nn.JoinTable(2, 2)({input, hiddenPrev})
    joinState2.data.module.name = 'joinState2'
    local state1 = hiddenDim * 2 + inputDim
    local state2 = hiddenDim + inputDim
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
    local stateNext = nn.JoinTable(2, 2)({cellNext, hiddenNext})
    stateNext.data.module.name = 'stateNext'
    local stackModule = nn.gModule({input, statePrev}, {stateNext})
    stackModule.moduleMap = {
        hiddenNext = hiddenNext.data.module
    }

    inGateLin.data.module.name = 'inGateLin'
    inTransformLin.data.module.name = 'inTransformLin'
    forgetGateLin.data.module.name = 'forgetGateLin'
    outGateLin.data.module.name = 'outGateLin'
    inGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.bias:fill(1)
    inTransformLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inTransformLin.data.module.bias:fill(0)
    forgetGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    forgetGateLin.data.module.bias:fill(1)
    outGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    outGateLin.data.module.bias:fill(1)
    return stackModule
end

-------------------------------------------------------------------------------
function lstm.createAttentionUnit(
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
    local attentionPrev = nn.Narrow(
        2, hiddenDim * 2 + itemDim + numItems + 1, numItems)(statePrev)

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
    local stayPenalty = nn.CMulTable()({transitionOut, attentionPrev})
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
        {cellNext, hiddenNext,
            attendedItems, attentionNorm,
            attentionAccum, attentionNormSel})
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
function lstm.createAttentionUnitDebug(
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
    local attention = nn.Reshape(numItems)(mm1)
    attention.data.module.name = 'attention'
    local attentionNorm = nn.SoftMax()(attention)

    local attentionReshape = nn.Reshape(1, numItems)(attention)
    local mm2 = nn.MM()({attentionReshape, items})
    mm2.data.module.name = 'MM2'
    local attendedItems = nn.Reshape(itemDim)(mm2)

    local joinState1 = nn.JoinTable(2, 2)({input, attendedItems, statePrev})
    joinState1.name = 'attJoinState1'
    local joinState2 = nn.JoinTable(2, 2)({input, attendedItems, hiddenPrev})
    joinState2.name = 'attJoinState2'
    local state1 = hiddenDim * 2 + inputDim + itemDim + numItems
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
    local stateNext = nn.JoinTable(2, 2)({cellNext, hiddenNext, attentionNorm})
    stateNext.name = 'attStateNext'
    local coreModule = nn.gModule({input, statePrev, items}, {stateNext})
    coreModule.moduleMap = {
        softAttention = attentionNorm.data.module
    -- attention = attention.data.module
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
return lstm

